import os
import json
import uuid
import random
from typing import TypedDict, Optional

import firebase_admin
from firebase_admin import firestore
from firebase_functions import https_fn, options
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph_checkpoint_firestore import FirestoreSaver

ACK_MESSAGES = [
    "Got it! I'm passing this to your care navigator right now. We'll get back to you shortly.",
    "Received! Your navigator has been notified and will follow up with you soon.",
    "On it! Connecting you with your care team now — hang tight.",
    "Thanks for reaching out. Your navigator is on it and will be in touch soon.",
    "Message received! We're looking into this for you right now.",
]

URGENT_ACK = "This sounds urgent — I'm flagging it for your care team right now. Someone will be with you very shortly."

SUPERVISOR_PROMPT = """You are an AI assistant for Arul, a care coordination platform for cancer patients.
A patient has sent a message. Prepare a structured task for their patient navigator.

Patient name: {patient_name}
Cancer type: {cancer_type}
Treatment phase: {treatment_phase}
Message: "{raw_message}"

Respond ONLY with a valid JSON object (no markdown, no extra text):
{{
  "taskType": one of ["appointment","medication","insurance","transportation","meal","emotional_support","general","unclear"],
  "urgency": one of ["urgent","routine","low"],
  "summary": "one clear sentence: what the patient needs",
  "navigatorNotes": "1-3 actionable sentences the navigator should know before responding",
  "extractedEntities": {{"dates": [], "locations": [], "people": [], "organizations": [], "other": []}}
}}
urgent=same-day/pain/emergency, routine=scheduled/standard, low=general/non-urgent, unclear=too ambiguous"""


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


# ── Lazy singletons ──────────────────────────────────────────────────────────
# Nothing initialises at import time — Firebase isn't ready then.
# Everything is created on first request and reused across warm invocations.

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


def get_graph():
    global _graph
    if _graph is not None:
        return _graph
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
    with FirestoreSaver.from_conn_info(
        project_id=os.environ.get("GCLOUD_PROJECT"),
        checkpoints_collection="lg_checkpoints",
        writes_collection="lg_writes",
    ) as checkpointer:
        _graph = build_graph(checkpointer)
    return _graph


# ── Nodes ────────────────────────────────────────────────────────────────────

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


def write_task_node(state: ArulState) -> dict:
    db = get_db()
    now = firestore.SERVER_TIMESTAMP
    msg_id = state["message_id"]

    db.collection("messages").document(msg_id).set({
        "messageId":      msg_id,
        "conversationId": state["conversation_id"],
        "patientId":      state["patient_id"],
        "sender":         "patient",
        "body":           state["raw_message"],
        "replyToMessageId": state.get("reply_to_message_id") or None,
        "taskId":         state["task_id"],
        "createdAt":      now,
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
        "createdAt":           now,
        "updatedAt":           now,
    })
    db.collection("conversations").document(state["conversation_id"]).set(
        {"lastActivity": now, "lastMessage": state["raw_message"], "status": "awaiting_navigator"},
        merge=True,
    )
    ack = URGENT_ACK if state["urgency"] == "urgent" else random.choice(ACK_MESSAGES)
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
        action = response.get("action", "resolve")
        message = response.get("message", "")
    else:
        action = "resolve"
        message = response
    return {"navigator_response": message, "navigator_action": action}


def send_followup_node(state: ArulState) -> dict:
    """Navigator asked a follow-up question. Send it to the patient and wait for their reply."""
    db = get_db()
    now = firestore.SERVER_TIMESTAMP
    question = (state.get("navigator_response") or "").strip()

    # Persist the navigator's follow-up question as a message
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

    # Mark task as awaiting patient response
    db.collection("tasks").document(state["task_id"]).update({
        "status":            "awaiting_patient",
        "navigatorResponse": question,
        "replyMessageId":    followup_msg_id,
        "updatedAt":         now,
    })
    db.collection("conversations").document(state["conversation_id"]).set(
        {"status": "awaiting_patient", "lastActivity": now},
        merge=True,
    )

    # Interrupt - wait for the patient to reply
    patient_reply = interrupt({
        "waiting_for":       "patient_reply",
        "taskId":            state["task_id"],
        "followupMessageId": followup_msg_id,
        "question":          question,
    })

    # ── Patient has replied — persist and re-surface for navigator ──
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
        "status":           "pending",
        "rawMessage":       patient_reply,
        "patientReply":     patient_reply,
        "imessageDelivered": False,
        "updatedAt":        now2,
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
    """Route to followup loop or final reply based on navigator action."""
    if state.get("navigator_action") == "followup":
        return "send_followup"
    return "format_reply"


def format_reply_node(state: ArulState) -> dict:
    db = get_db()
    reply = (state.get("navigator_response") or "").strip() or "Your navigator will follow up shortly."
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
        "status":             "completed",
        "navigatorResponse":  state["navigator_response"],
        "replyMessageId":     reply_msg_id,
        "updatedAt":          now,
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
    g.add_node("write_task",         write_task_node)
    g.add_node("wait_for_navigator", wait_for_navigator_node)
    g.add_node("send_followup",      send_followup_node)
    g.add_node("format_reply",       format_reply_node)
    g.add_edge(START,                "intake")
    g.add_edge("intake",             "supervisor")
    g.add_edge("supervisor",         "write_task")
    g.add_edge("write_task",         "wait_for_navigator")
    g.add_conditional_edges("wait_for_navigator", route_after_navigator,
                            {"send_followup": "send_followup", "format_reply": "format_reply"})
    g.add_edge("send_followup",      "wait_for_navigator")  # loop back
    g.add_edge("format_reply",       END)
    return g.compile(checkpointer=checkpointer)


# ── Endpoints ────────────────────────────────────────────────────────────────

@https_fn.on_request(
    region="us-central1",
    memory=options.MemoryOption.MB_512,
    timeout_sec=120,
    min_instances=0,
    max_instances=10,
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
        if not all([patient_id, conversation_id, raw_message]):
            return https_fn.Response(
                json.dumps({"error": "Missing: patientId, conversationId, message"}),
                status=400, mimetype="application/json",
            )
        message_id = str(uuid.uuid4())
        graph = get_graph()
        result = graph.invoke(
            {
                "raw_message":          raw_message,
                "patient_id":           patient_id,
                "conversation_id":      conversation_id,
                "patient_name":         "",
                "patient_context":      {},
                "task_id":              "",
                "task_type":            "unclear",
                "urgency":              "routine",
                "summary":              "",
                "navigator_notes":      "",
                "extracted_entities":   {},
                "ack_message":          "",
                "navigator_response":   None,
                "navigator_action":     None,
                "final_message":        "",
                "message_id":           message_id,
                "reply_to_message_id":  None,
            },
            {"configurable": {"thread_id": conversation_id}},
        )
        return https_fn.Response(
            json.dumps({
                "success":    True,
                "messageId":  result["message_id"],
                "ackMessage": result["ack_message"],
                "taskId":     result["task_id"],
                "urgency":    result["urgency"],
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
    min_instances=0,
    max_instances=10,
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
        resume_data = {"action": action, "message": navigator_response}
        result = graph.invoke(
            Command(resume=resume_data),
            {"configurable": {"thread_id": conversation_id}},
        )
        if action == "followup":
            return https_fn.Response(
                json.dumps({
                    "success": True,
                    "status":  "awaiting_patient",
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
    min_instances=0,
    max_instances=10,
    secrets=["GOOGLE_API_KEY"],
    cors=options.CorsOptions(cors_origins=["*"], cors_methods=["POST"]),
)
def patient_reply(req: https_fn.Request) -> https_fn.Response:
    """Resume the graph when a patient replies to a navigator's follow-up question."""
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