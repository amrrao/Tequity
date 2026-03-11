import os
import json
import uuid
from typing import TypedDict, Optional

import firebase_admin
from firebase_admin import firestore
from firebase_functions import https_fn, options
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph_checkpoint_firestore import FirestoreSaver

URGENT_ACK = "This sounds urgent — I'm flagging it for your care team right now. Someone will be with you very shortly."

SUPERVISOR_PROMPT = """You are an AI assistant for Arul, a care coordination platform for cancer patients.
A patient has sent a message. Determine whether this is a care need or casual conversation.

Patient name: {patient_name}
Cancer type: {cancer_type}
Treatment phase: {treatment_phase}
Message: "{raw_message}"

Respond ONLY with a valid JSON object (no markdown, no extra text):
{{
  "taskType": one of ["appointment","medication","insurance","transportation","meal","emotional_support","general","unclear","chitchat"],
  "urgency": one of ["urgent","routine","low"],
  "summary": "one clear sentence: what the patient needs or said",
  "navigatorNotes": "1-3 actionable sentences the navigator should know before responding",
  "extractedEntities": {{"dates": [], "locations": [], "people": [], "organizations": [], "other": []}}
}}

taskType rules:
- "chitchat": greetings, small talk, thank yous, acknowledgments, emoji-only, or any message with NO actionable care need
- "emotional_support": patient is expressing distress, fear, or anxiety — even if phrased casually — this is NOT chitchat
- urgent=same-day/pain/emergency, routine=scheduled/standard, low=general/non-urgent, unclear=too ambiguous
- When in doubt between chitchat and a real need, choose the real need taskType"""

CHITCHAT_PROMPT = """You are a warm, caring presence at Arul — a care coordination platform for cancer patients.
A patient has sent you a casual message. Respond naturally and warmly.

Patient name: {patient_name}
Cancer type: {cancer_type}
Treatment phase: {treatment_phase}
Message: "{raw_message}"

Guidelines:
- Address them by their preferred name
- Warm, human, conversational — never clinical or robotic
- Keep it brief: 1-3 sentences max
- Do NOT mention AI, bots, or automation
- Do NOT use the word "navigator"
- If they say thank you, acknowledge it warmly and remind them you're always here
- If they say hi or hello, greet them back and gently invite them to reach out if they need anything
- Never make up medical information or appointments

Respond with ONLY the message text, no quotes, no preamble."""

ACK_PROMPT = """You are a warm, caring presence at Arul — a care coordination platform for cancer patients.
A patient just sent a message that needs attention from their care team. Write a brief acknowledgment SMS.

Patient name: {patient_name}
Cancer type: {cancer_type}
Treatment phase: {treatment_phase}
Task type: {task_type}
Patient message: "{raw_message}"

Guidelines:
- Address them by their preferred name
- Warm and reassuring — never clinical or robotic
- Let them know their message was received and someone will follow up soon
- 1-2 sentences max
- Do NOT mention AI, bots, or automation
- Do NOT use the word "navigator" — say "care team" or "we"
- Do NOT promise a specific timeframe unless urgency is urgent

Respond with ONLY the message text, no quotes, no preamble."""

WELCOME_PROMPT = """You are a care navigator at Arul, a warm and human-centered cancer care coordination platform.

Write a short, personal welcome text message to a newly onboarded patient. This is the very first message they will receive from their care team.

Patient name: {patient_name}
Cancer type: {cancer_type}
Treatment phase: {treatment_phase}
Navigator name: {navigator_name}

Guidelines:
- Address them by their preferred name
- Warm, human, never clinical or robotic
- Mention their navigator by first name
- Let them know they can text anytime with questions or needs
- 3-4 sentences max
- Do NOT mention AI, bots, or automation
- Do NOT use the word "navigator" — say "care team" or use the navigator's name directly
- End with something that invites them to reach out

Respond with ONLY the message text, no quotes, no preamble."""

CLASSIFIER_PROMPT = """You are a routing assistant for Arul, a care coordination platform for cancer patients.

A patient sent a new message. They may have open care conversations.

New message: "{new_message}"

Open conversations:
{open_conversations}

Does this message continue one of the open conversations, or is it a new issue?

Respond ONLY with valid JSON:
{{
  "decision": "existing" or "new",
  "conversationId": "<id if existing, else null>",
  "confidence": "high" or "low",
  "reasoning": "one sentence"
}}

Rules:
- "existing" only if the message clearly and directly continues an open conversation topic
- "new" if it introduces a different issue, even if related
- "low" confidence if genuinely ambiguous — when in doubt use "new"
- Never return "existing" with low confidence"""


class ArulState(TypedDict):
    raw_message: str
    patient_id: str
    conversation_id: str
    patient_name: str
    patient_context: dict
    task_id: str
    task_type: str
    urgency: str
    summary: str
    navigator_notes: str
    extracted_entities: dict
    ack_message: str
    navigator_response: Optional[str]
    navigator_action: Optional[str]
    final_message: str
    message_id: str
    reply_to_message_id: Optional[str]
    is_ambiguous: bool


_db = None
_llm = None
_graph = None


def get_db():
    global _db
    if _db is None:
        if not firebase_admin._apps:
            firebase_admin.initialize_app()
        _db = firestore.client()
    return _db


def get_llm():
    global _llm
    if _llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            temperature=0.1,
        )
    return _llm


def get_llm_warm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0.5,
    )


def get_graph():
    global _graph
    if _graph is not None:
        return _graph
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
    checkpointer = FirestoreSaver(
        project_id=os.environ.get("GCLOUD_PROJECT"),
        checkpoints_collection="lg_checkpoints",
        writes_collection="lg_writes",
    )
    _graph = build_graph(checkpointer)
    return _graph


def intake_node(state: ArulState) -> dict:
    db = get_db()
    doc = db.collection("patients").document(state["patient_id"]).get()
    data = doc.to_dict() if doc.exists else {}
    return {
        "patient_name": data.get("preferredName") or data.get("name") or "there",
        "patient_context": {
            "cancer_type":     data.get("cancerType", ""),
            "treatment_phase": data.get("treatmentPhase", ""),
            "navigator_id":    data.get("navigatorId", ""),
        },
    }


def supervisor_node(state: ArulState) -> dict:
    from langchain_core.messages import SystemMessage, HumanMessage
    llm = get_llm()
    ctx = state.get("patient_context", {})
    prompt = SUPERVISOR_PROMPT.format(
        patient_name=state["patient_name"],
        cancer_type=ctx.get("cancer_type") or "unknown",
        treatment_phase=ctx.get("treatment_phase") or "unknown",
        raw_message=state["raw_message"],
    )
    response = llm.invoke([
        SystemMessage(content="You are a medical care coordination AI. Respond only with valid JSON."),
        HumanMessage(content=prompt),
    ])
    raw = response.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {
            "taskType": "unclear",
            "urgency": "routine",
            "summary": "Could not parse patient message automatically.",
            "navigatorNotes": f"Please review the raw message: {state['raw_message']}",
            "extractedEntities": {"dates": [], "locations": [], "people": [], "organizations": [], "other": []},
        }
    return {
        "task_id":            str(uuid.uuid4()),
        "task_type":          parsed.get("taskType", "unclear"),
        "urgency":            parsed.get("urgency", "routine"),
        "summary":            parsed.get("summary", ""),
        "navigator_notes":    parsed.get("navigatorNotes", ""),
        "extracted_entities": parsed.get("extractedEntities", {}),
    }


def route_after_supervisor(state: ArulState) -> str:
    return "ai_reply" if state.get("task_type") == "chitchat" else "write_task"


def ai_reply_node(state: ArulState) -> dict:
    ctx = state.get("patient_context", {})
    prompt = CHITCHAT_PROMPT.format(
        patient_name=state["patient_name"],
        cancer_type=ctx.get("cancer_type") or "unknown",
        treatment_phase=ctx.get("treatment_phase") or "unknown",
        raw_message=state["raw_message"],
    )
    response = get_llm_warm().invoke(prompt)
    return {"ack_message": response.content.strip()}


def write_task_node(state: ArulState) -> dict:
    db = get_db()
    now = firestore.SERVER_TIMESTAMP
    msg_id = state["message_id"]

    conv_doc = db.collection("conversations").document(state["conversation_id"]).get()
    is_first_message = not conv_doc.exists

    db.collection("messages").document(msg_id).set({
        "messageId":        msg_id,
        "conversationId":   state["conversation_id"],
        "patientId":        state["patient_id"],
        "sender":           "patient",
        "body":             state["raw_message"],
        "replyToMessageId": state.get("reply_to_message_id") or None,
        "taskId":           state["task_id"],
        "createdAt":        now,
    })

    db.collection("tasks").document(state["task_id"]).set({
        "taskId":              state["task_id"],
        "patientId":           state["patient_id"],
        "conversationId":      state["conversation_id"],
        "messageId":           msg_id,
        "patientName":         state["patient_name"],
        "rawMessage":          state["raw_message"],
        "taskType":            state["task_type"],
        "urgency":             state["urgency"],
        "summary":             state["summary"],
        "navigatorNotes":      state["navigator_notes"],
        "extractedEntities":   state["extracted_entities"],
        "status":              "pending",
        "assignedNavigatorId": state["patient_context"].get("navigator_id") or None,
        "navigatorResponse":   None,
        "ambiguous":           state.get("is_ambiguous", False),
        "imessageDelivered":   False,
        "createdAt":           now,
        "updatedAt":           now,
    })

    db.collection("conversations").document(state["conversation_id"]).set(
        {
            "lastActivity": now,
            "lastMessage":  state["raw_message"],
            "status":       "awaiting_navigator",
            "patientId":    state["patient_id"],
        },
        merge=True,
    )

    ack = ""
    if is_first_message:
        try:
            ctx = state.get("patient_context", {})
            if state["urgency"] == "urgent":
                ack = URGENT_ACK
            else:
                prompt = ACK_PROMPT.format(
                    patient_name=state["patient_name"],
                    cancer_type=ctx.get("cancer_type") or "unknown",
                    treatment_phase=ctx.get("treatment_phase") or "unknown",
                    task_type=state["task_type"],
                    raw_message=state["raw_message"],
                )
                response = get_llm_warm().invoke(prompt)
                ack = response.content.strip()
        except Exception:
            ack = "Got it! Your care team has been notified and will follow up with you soon."

    return {"ack_message": ack}


def wait_for_navigator_node(state: ArulState) -> dict:
    response = interrupt({
        "waiting_for": "navigator_response",
        "taskId":      state["task_id"],
        "messageId":   state["message_id"],
        "patientName": state["patient_name"],
        "summary":     state["summary"],
    })
    if isinstance(response, dict):
        action  = response.get("action", "resolve")
        message = response.get("message", "")
    else:
        action  = "resolve"
        message = response
    return {"navigator_response": message, "navigator_action": action}


def send_followup_node(state: ArulState) -> dict:
    db = get_db()
    now = firestore.SERVER_TIMESTAMP
    question = (state.get("navigator_response") or "").strip()

    followup_msg_id = str(uuid.uuid4())
    db.collection("messages").document(followup_msg_id).set({
        "messageId":        followup_msg_id,
        "conversationId":   state["conversation_id"],
        "patientId":        state["patient_id"],
        "sender":           "navigator",
        "body":             question,
        "replyToMessageId": state["message_id"],
        "taskId":           state["task_id"],
        "createdAt":        now,
    })

    db.collection("tasks").document(state["task_id"]).update({
        "status":            "awaiting_patient",
        "navigatorResponse": question,
        "replyMessageId":    followup_msg_id,
        "imessageDelivered": False,
        "updatedAt":         now,
    })
    db.collection("conversations").document(state["conversation_id"]).set(
        {"status": "awaiting_patient", "lastActivity": now},
        merge=True,
    )

    patient_reply = interrupt({
        "waiting_for":       "patient_reply",
        "taskId":            state["task_id"],
        "followupMessageId": followup_msg_id,
        "question":          question,
    })

    new_msg_id = str(uuid.uuid4())
    now2 = firestore.SERVER_TIMESTAMP

    db.collection("messages").document(new_msg_id).set({
        "messageId":        new_msg_id,
        "conversationId":   state["conversation_id"],
        "patientId":        state["patient_id"],
        "sender":           "patient",
        "body":             patient_reply,
        "replyToMessageId": followup_msg_id,
        "taskId":           state["task_id"],
        "createdAt":        now2,
    })

    db.collection("tasks").document(state["task_id"]).update({
        "status":            "pending",
        "rawMessage":        patient_reply,
        "patientReply":      patient_reply,
        "imessageDelivered": False,
        "updatedAt":         now2,
    })
    db.collection("conversations").document(state["conversation_id"]).set(
        {"status": "awaiting_navigator", "lastActivity": now2, "lastMessage": patient_reply},
        merge=True,
    )

    return {
        "raw_message":        patient_reply,
        "message_id":         new_msg_id,
        "navigator_response": None,
        "navigator_action":   None,
    }


def route_after_navigator(state: ArulState) -> str:
    return "send_followup" if state.get("navigator_action") == "followup" else "format_reply"


def format_reply_node(state: ArulState) -> dict:
    db = get_db()
    reply = (state.get("navigator_response") or "").strip() or "Your care team will follow up shortly."
    now = firestore.SERVER_TIMESTAMP

    reply_msg_id = str(uuid.uuid4())
    db.collection("messages").document(reply_msg_id).set({
        "messageId":        reply_msg_id,
        "conversationId":   state["conversation_id"],
        "patientId":        state["patient_id"],
        "sender":           "navigator",
        "body":             reply,
        "replyToMessageId": state["message_id"],
        "taskId":           state["task_id"],
        "createdAt":        now,
    })

    db.collection("tasks").document(state["task_id"]).update({
        "status":            "completed",
        "navigatorResponse": state["navigator_response"],
        "replyMessageId":    reply_msg_id,
        "imessageDelivered": False,
        "updatedAt":         now,
    })
    db.collection("conversations").document(state["conversation_id"]).set(
        {"status": "resolved", "lastActivity": now},
        merge=True,
    )
    return {"final_message": reply, "reply_to_message_id": state["message_id"]}


def build_graph(checkpointer):
    g = StateGraph(ArulState)
    g.add_node("intake",             intake_node)
    g.add_node("supervisor",         supervisor_node)
    g.add_node("ai_reply",           ai_reply_node)
    g.add_node("write_task",         write_task_node)
    g.add_node("wait_for_navigator", wait_for_navigator_node)
    g.add_node("send_followup",      send_followup_node)
    g.add_node("format_reply",       format_reply_node)
    g.add_edge(START,                "intake")
    g.add_edge("intake",             "supervisor")
    g.add_conditional_edges("supervisor", route_after_supervisor,
                            {"ai_reply": "ai_reply", "write_task": "write_task"})
    g.add_edge("ai_reply",           END)
    g.add_edge("write_task",         "wait_for_navigator")
    g.add_conditional_edges("wait_for_navigator", route_after_navigator,
                            {"send_followup": "send_followup", "format_reply": "format_reply"})
    g.add_edge("send_followup",      "wait_for_navigator")
    g.add_edge("format_reply",       END)
    return g.compile(checkpointer=checkpointer)


@https_fn.on_request(
    region="us-central1",
    memory=options.MemoryOption.MB_512,
    timeout_sec=120,
    cpu=1,
    min_instances=0,
    max_instances=10,
    concurrency=80,
    secrets=["GOOGLE_API_KEY"],
    cors=options.CorsOptions(cors_origins=["*"], cors_methods=["POST"]),
)
def message(req: https_fn.Request) -> https_fn.Response:
    if req.method != "POST":
        return https_fn.Response("Method not allowed", status=405)
    try:
        body            = req.get_json(silent=True) or {}
        patient_id      = str(body.get("patientId", "")).strip()
        conversation_id = str(body.get("conversationId", "")).strip()
        raw_message     = str(body.get("message", "")).strip()
        is_ambiguous    = bool(body.get("isAmbiguous", False))
        if not all([patient_id, conversation_id, raw_message]):
            return https_fn.Response(
                json.dumps({"error": "Missing: patientId, conversationId, message"}),
                status=400, mimetype="application/json",
            )
        message_id = str(uuid.uuid4())
        graph = get_graph()
        result = graph.invoke(
            {
                "raw_message":         raw_message,
                "patient_id":          patient_id,
                "conversation_id":     conversation_id,
                "patient_name":        "",
                "patient_context":     {},
                "task_id":             "",
                "task_type":           "unclear",
                "urgency":             "routine",
                "summary":             "",
                "navigator_notes":     "",
                "extracted_entities":  {},
                "ack_message":         "",
                "navigator_response":  None,
                "navigator_action":    None,
                "final_message":       "",
                "message_id":          message_id,
                "reply_to_message_id": None,
                "is_ambiguous":        is_ambiguous,
            },
            {"configurable": {"thread_id": conversation_id}},
        )
        return https_fn.Response(
            json.dumps({
                "success":    True,
                "messageId":  result["message_id"],
                "ackMessage": result["ack_message"],
                "taskId":     result.get("task_id", ""),
                "urgency":    result.get("urgency", "low"),
                "isChitchat": result.get("task_type") == "chitchat",
            }),
            status=200, mimetype="application/json",
        )
    except Exception as e:
        return https_fn.Response(
            json.dumps({"success": False, "error": str(e)}),
            status=500, mimetype="application/json",
        )


@https_fn.on_request(
    region="us-central1",
    memory=options.MemoryOption.MB_512,
    timeout_sec=120,
    cpu=1,
    min_instances=0,
    max_instances=10,
    concurrency=80,
    secrets=["GOOGLE_API_KEY"],
    cors=options.CorsOptions(cors_origins=["*"], cors_methods=["POST"]),
)
def navigator_reply(req: https_fn.Request) -> https_fn.Response:
    if req.method != "POST":
        return https_fn.Response("Method not allowed", status=405)
    try:
        body               = req.get_json(silent=True) or {}
        conversation_id    = str(body.get("conversationId", "")).strip()
        navigator_response = str(body.get("navigatorResponse", "")).strip()
        action             = str(body.get("action", "resolve")).strip()
        reply_to_msg_id    = str(body.get("replyToMessageId", "")).strip() or None
        if not all([conversation_id, navigator_response]):
            return https_fn.Response(
                json.dumps({"error": "Missing: conversationId, navigatorResponse"}),
                status=400, mimetype="application/json",
            )
        graph = get_graph()
        result = graph.invoke(
            Command(resume={"action": action, "message": navigator_response}),
            {"configurable": {"thread_id": conversation_id}},
        )
        if action == "followup":
            return https_fn.Response(
                json.dumps({
                    "success":         True,
                    "status":          "awaiting_patient",
                    "followupMessage": navigator_response,
                }),
                status=200, mimetype="application/json",
            )
        return https_fn.Response(
            json.dumps({
                "success":          True,
                "status":           "completed",
                "finalMessage":     result["final_message"],
                "replyToMessageId": result.get("reply_to_message_id") or reply_to_msg_id,
            }),
            status=200, mimetype="application/json",
        )
    except Exception as e:
        return https_fn.Response(
            json.dumps({"success": False, "error": str(e)}),
            status=500, mimetype="application/json",
        )


@https_fn.on_request(
    region="us-central1",
    memory=options.MemoryOption.MB_512,
    timeout_sec=120,
    cpu=1,
    min_instances=0,
    max_instances=10,
    concurrency=80,
    secrets=["GOOGLE_API_KEY"],
    cors=options.CorsOptions(cors_origins=["*"], cors_methods=["POST"]),
)
def patient_reply(req: https_fn.Request) -> https_fn.Response:
    if req.method != "POST":
        return https_fn.Response("Method not allowed", status=405)
    try:
        body            = req.get_json(silent=True) or {}
        conversation_id = str(body.get("conversationId", "")).strip()
        raw_message     = str(body.get("message", "")).strip()
        patient_id      = str(body.get("patientId", "")).strip()
        if not all([conversation_id, raw_message, patient_id]):
            return https_fn.Response(
                json.dumps({"error": "Missing: conversationId, message, patientId"}),
                status=400, mimetype="application/json",
            )
        graph = get_graph()
        result = graph.invoke(
            Command(resume=raw_message),
            {"configurable": {"thread_id": conversation_id}},
        )
        return https_fn.Response(
            json.dumps({
                "success":   True,
                "status":    "awaiting_navigator",
                "messageId": result.get("message_id"),
                "taskId":    result.get("task_id"),
            }),
            status=200, mimetype="application/json",
        )
    except Exception as e:
        return https_fn.Response(
            json.dumps({"success": False, "error": str(e)}),
            status=500, mimetype="application/json",
        )


@https_fn.on_request(
    region="us-central1",
    memory=options.MemoryOption.MB_512,
    timeout_sec=60,
    cpu=1,
    min_instances=0,
    max_instances=10,
    concurrency=80,
    secrets=["GOOGLE_API_KEY"],
    cors=options.CorsOptions(cors_origins=["*"], cors_methods=["GET", "POST", "OPTIONS"]),
)
def onboard_patient(req: https_fn.Request) -> https_fn.Response:
    if req.method != "POST":
        return https_fn.Response("Method not allowed", status=405)
    try:
        body           = req.get_json(silent=True) or {}
        patient_id     = str(body.get("patientId", "")).strip()
        navigator_name = str(body.get("navigatorName", "your care team")).strip()
        if not patient_id:
            return https_fn.Response(
                json.dumps({"error": "Missing: patientId"}),
                status=400, mimetype="application/json",
            )
        db = get_db()
        doc = db.collection("patients").document(patient_id).get()
        if not doc.exists:
            return https_fn.Response(
                json.dumps({"error": "Patient not found"}),
                status=404, mimetype="application/json",
            )
        data            = doc.to_dict()
        patient_name    = data.get("preferredName") or data.get("name") or "there"
        cancer_type     = data.get("cancerType") or "cancer"
        treatment_phase = data.get("treatmentPhase") or "treatment"
        prompt = WELCOME_PROMPT.format(
            patient_name=patient_name,
            cancer_type=cancer_type,
            treatment_phase=treatment_phase,
            navigator_name=navigator_name,
        )
        response = get_llm_warm().invoke(prompt)
        welcome_message = response.content.strip()
        now             = firestore.SERVER_TIMESTAMP
        conversation_id = f"conv_onboard_{patient_id}"
        db.collection("conversations").document(conversation_id).set({
            "conversationId":   conversation_id,
            "patientId":        patient_id,
            "status":           "awaiting_patient",
            "lastMessage":      welcome_message,
            "lastActivity":     now,
            "welcomeDelivered": False,
        })
        msg_id = str(uuid.uuid4())
        db.collection("messages").document(msg_id).set({
            "messageId":      msg_id,
            "conversationId": conversation_id,
            "patientId":      patient_id,
            "sender":         "navigator",
            "body":           welcome_message,
            "createdAt":      now,
        })
        return https_fn.Response(
            json.dumps({
                "success":        True,
                "welcomeMessage": welcome_message,
                "conversationId": conversation_id,
                "messageId":      msg_id,
            }),
            status=200, mimetype="application/json",
        )
    except Exception as e:
        return https_fn.Response(
            json.dumps({"success": False, "error": str(e)}),
            status=500, mimetype="application/json",
        )


@https_fn.on_request(
    region="us-central1",
    memory=options.MemoryOption.MB_512,
    timeout_sec=30,
    cpu=1,
    min_instances=0,
    max_instances=10,
    concurrency=80,
    secrets=["GOOGLE_API_KEY"],
    cors=options.CorsOptions(cors_origins=["*"], cors_methods=["POST"]),
)
def classify_message(req: https_fn.Request) -> https_fn.Response:
    if req.method != "POST":
        return https_fn.Response("Method not allowed", status=405)
    try:
        body       = req.get_json(silent=True) or {}
        patient_id = str(body.get("patientId", "")).strip()
        raw_msg    = str(body.get("message", "")).strip()
        if not all([patient_id, raw_msg]):
            return https_fn.Response(
                json.dumps({"error": "Missing: patientId, message"}),
                status=400, mimetype="application/json",
            )
        db = get_db()
        tasks_snap = (
            db.collection("tasks")
            .where("patientId", "==", patient_id)
            .where("status", "in", ["pending", "awaiting_patient"])
            .stream()
        )
        open_convs = []
        for t in tasks_snap:
            td = t.to_dict()
            if td.get("summary"):
                open_convs.append(
                    f"- conversationId: {td['conversationId']} | issue: {td['summary']}"
                )
        if not open_convs:
            return https_fn.Response(
                json.dumps({"decision": "new", "conversationId": None, "confidence": "high", "reasoning": "No open conversations"}),
                status=200, mimetype="application/json",
            )
        llm = get_llm()
        prompt = CLASSIFIER_PROMPT.format(
            new_message=raw_msg,
            open_conversations="\n".join(open_convs),
        )
        response = llm.invoke(prompt)
        raw = response.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"decision": "new", "conversationId": None, "confidence": "low", "reasoning": "Parse error"}
        if result.get("confidence") == "low":
            result["decision"] = "new"
            result["conversationId"] = None
        return https_fn.Response(
            json.dumps(result),
            status=200, mimetype="application/json",
        )
    except Exception as e:
        return https_fn.Response(
            json.dumps({"success": False, "error": str(e)}),
            status=500, mimetype="application/json",
        )