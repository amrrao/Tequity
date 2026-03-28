import os
import json
import uuid
from firebase_admin import firestore
from firebase_functions import https_fn, options
from langgraph.types import Command

from gmail_oauth import generate_gmail_oauth_url, gmail_oauth_callback, send_gmail_on_behalf
from services import get_db, get_llm, get_llm_warm, get_graph
from prompts import WELCOME_PROMPT, CLASSIFIER_PROMPT


def _check_dedup(db, dedup_key):
    dedup_ref = db.collection("dedup").document(dedup_key)
    dedup_doc = dedup_ref.get()
    if dedup_doc.exists:
        data = dedup_doc.to_dict()
        if data.get("status") == "completed":
            return data.get("response")
    return None


def _set_dedup_processing(db, dedup_key):
    dedup_ref = db.collection("dedup").document(dedup_key)
    dedup_ref.set({"status": "processing", "createdAt": firestore.SERVER_TIMESTAMP})


def _set_dedup_completed(db, dedup_key, response_data):
    dedup_ref = db.collection("dedup").document(dedup_key)
    dedup_ref.set({
        "status": "completed",
        "response": response_data,
        "completedAt": firestore.SERVER_TIMESTAMP
    }, merge=True)


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
        db = get_db()
        dedup_key = f"msg_{conversation_id}_{message_id}"
        
        cached = _check_dedup(db, dedup_key)
        if cached:
            return https_fn.Response(
                json.dumps(cached),
                status=200, mimetype="application/json",
            )
        
        _set_dedup_processing(db, dedup_key)
        
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
                "is_outbound":         False,
                "route":               "care_need",
            },
            {"configurable": {"thread_id": conversation_id}},
        )
        
        response_data = {
            "success":    True,
            "messageId":  result["message_id"],
            "ackMessage": result["ack_message"],
            "taskId":     result.get("task_id", ""),
            "urgency":    result.get("urgency", "low"),
            "isChitchat": result.get("task_type") == "chitchat",
        }
        
        _set_dedup_completed(db, dedup_key, response_data)
        
        return https_fn.Response(
            json.dumps(response_data),
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
        
        db = get_db()
        dedup_key = f"nav_{conversation_id}_{hash(navigator_response)}"
        
        cached = _check_dedup(db, dedup_key)
        if cached:
            return https_fn.Response(
                json.dumps(cached),
                status=200, mimetype="application/json",
            )
        
        _set_dedup_processing(db, dedup_key)
        
        graph = get_graph()
        result = graph.invoke(
            Command(resume={"action": action, "message": navigator_response}),
            {"configurable": {"thread_id": conversation_id}},
        )
        
        if action == "followup":
            response_data = {
                "success":         True,
                "status":          "awaiting_patient",
                "followupMessage": navigator_response,
            }
        else:
            response_data = {
                "success":          True,
                "status":           "completed",
                "finalMessage":     result["final_message"],
                "replyToMessageId": result.get("reply_to_message_id") or reply_to_msg_id,
            }
        
        _set_dedup_completed(db, dedup_key, response_data)
        
        return https_fn.Response(
            json.dumps(response_data),
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
        
        db = get_db()
        
        task_snap = list(
            db.collection("tasks")
            .where("conversationId", "==", conversation_id)
            .where("status", "==", "awaiting_patient")
            .limit(1)
            .stream()
        )
        if not task_snap:
            return https_fn.Response(
                json.dumps({"error": "No awaiting_patient task found for this conversation"}),
                status=409, mimetype="application/json",
            )
        
        dedup_key = f"pat_{conversation_id}_{hash(raw_message)}"
        
        cached = _check_dedup(db, dedup_key)
        if cached:
            return https_fn.Response(
                json.dumps(cached),
                status=200, mimetype="application/json",
            )
        
        _set_dedup_processing(db, dedup_key)
        
        graph = get_graph()
        result = graph.invoke(
            Command(resume=raw_message),
            {"configurable": {"thread_id": conversation_id}},
        )
        
        response_data = {
            "success":   True,
            "status":    "awaiting_navigator",
            "messageId": result.get("message_id"),
            "taskId":    result.get("task_id"),
        }
        
        _set_dedup_completed(db, dedup_key, response_data)
        
        return https_fn.Response(
            json.dumps(response_data),
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

        conversation_id = f"conv_onboard_{patient_id}"
        existing = db.collection("conversations").document(conversation_id).get()
        if existing.exists and existing.to_dict().get("welcomeDelivered"):
            return https_fn.Response(
                json.dumps({"error": "Patient already onboarded"}),
                status=409, mimetype="application/json",
            )

        prompt = WELCOME_PROMPT.format(
            patient_name=patient_name,
            cancer_type=cancer_type,
            treatment_phase=treatment_phase,
            navigator_name=navigator_name,
        )
        response = get_llm_warm().invoke(prompt)
        welcome_message = response.content.strip()
        now             = firestore.SERVER_TIMESTAMP

        batch = db.batch()

        conv_ref = db.collection("conversations").document(conversation_id)
        batch.set(conv_ref, {
            "conversationId":   conversation_id,
            "patientId":        patient_id,
            "status":           "awaiting_patient",
            "lastMessage":      welcome_message,
            "lastActivity":     now,
            "welcomeDelivered": False,
            "navigatorName":    navigator_name,
        })

        msg_id = str(uuid.uuid4())
        msg_ref = db.collection("messages").document(msg_id)
        batch.set(msg_ref, {
            "messageId":      msg_id,
            "conversationId": conversation_id,
            "patientId":      patient_id,
            "sender":         "navigator",
            "body":           welcome_message,
            "createdAt":      now,
        })

        batch.commit()

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
            .order_by("createdAt", direction=firestore.Query.DESCENDING)
            .limit(10)
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
def outbound_message(req: https_fn.Request) -> https_fn.Response:
    if req.method != "POST":
        return https_fn.Response("Method not allowed", status=405)
    try:
        body                  = req.get_json(silent=True) or {}
        patient_id            = str(body.get("patientId", "")).strip()
        conversation_id       = str(body.get("conversationId", "")).strip()
        raw_message           = str(body.get("message", "")).strip()
        navigator_id          = str(body.get("navigatorId", "")).strip()
        is_existing_conversation = bool(body.get("isExistingConversation", False))
        
        if not all([patient_id, conversation_id, raw_message, navigator_id]):
            return https_fn.Response(
                json.dumps({"error": "Missing: patientId, conversationId, message, navigatorId"}),
                status=400, mimetype="application/json",
            )

        db = get_db()
        message_id = str(uuid.uuid4())
        now = firestore.SERVER_TIMESTAMP

        if is_existing_conversation:
            batch = db.batch()

            msg_ref = db.collection("messages").document(message_id)
            batch.set(msg_ref, {
                "messageId":        message_id,
                "conversationId":   conversation_id,
                "patientId":        patient_id,
                "sender":           "navigator",
                "body":             raw_message,
                "replyToMessageId": None,
                "taskId":           None,
                "createdAt":        now,
            })

            task_id = str(uuid.uuid4())
            task_ref = db.collection("tasks").document(task_id)
            batch.set(task_ref, {
                "taskId":            task_id,
                "patientId":         patient_id,
                "conversationId":    conversation_id,
                "messageId":         message_id,
                "rawMessage":        raw_message,
                "status":            "completed",
                "navigatorResponse": raw_message,
                "replyMessageId":    message_id,
                "imessageDelivered": False,
                "isOutbound":        True,
                "taskType":          "general",
                "urgency":           "routine",
                "summary":           f"Navigator sent proactive message: {raw_message[:80]}",
                "navigatorNotes":    "",
                "extractedEntities": {},
                "ambiguous":         False,
                "createdAt":         now,
                "updatedAt":         now,
            })

            conv_ref = db.collection("conversations").document(conversation_id)
            batch.set(conv_ref, {
                "lastActivity": now,
                "lastMessage":  raw_message,
                "status":       "awaiting_patient",
            }, merge=True)

            batch.commit()

            return https_fn.Response(
                json.dumps({
                    "success":        True,
                    "messageId":      message_id,
                    "conversationId": conversation_id,
                    "taskId":         task_id,
                }),
                status=200, mimetype="application/json",
            )

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
                "is_ambiguous":        False,
                "is_outbound":         True,
                "route":               "outbound",
            },
            {"configurable": {"thread_id": conversation_id}},
        )
        return https_fn.Response(
            json.dumps({
                "success":        True,
                "messageId":      message_id,
                "conversationId": conversation_id,
                "taskId":         result.get("task_id", ""),
            }),
            status=200, mimetype="application/json",
        )
    except Exception as e:
        return https_fn.Response(
            json.dumps({"success": False, "error": str(e)}),
            status=500, mimetype="application/json",
        )
