import hashlib
import json
import logging
import uuid
from firebase_admin import firestore
from firebase_functions import https_fn, options
from langgraph.types import Command

# Imported so Firebase Functions discovers the decorated endpoints in gmail_oauth.py.
from gmail_oauth import generate_gmail_oauth_url, gmail_oauth_callback, send_gmail_on_behalf  # noqa: F401
from services import get_db, get_llm, get_llm_warm, get_graph
from prompts import WELCOME_PROMPT, CLASSIFIER_PROMPT

log = logging.getLogger(__name__)

# Shared Cloud Function configuration.
_FN_OPTS = dict(
    region="us-central1",
    memory=options.MemoryOption.MB_512,
    cpu=1,
    min_instances=0,
    max_instances=10,
    concurrency=80,
    secrets=["GOOGLE_API_KEY"],
    cors=options.CorsOptions(cors_origins=["*"], cors_methods=["POST"]),
)


def _json_response(data, status=200):
    return https_fn.Response(json.dumps(data), status=status, mimetype="application/json")


def _error_response(msg, status=400):
    return _json_response({"error": msg}, status=status)


def _dedup_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _check_dedup(db, dedup_key):
    doc = db.collection("dedup").document(dedup_key).get()
    if doc.exists:
        data = doc.to_dict()
        if data.get("status") == "completed":
            return data.get("response")
    return None


def _set_dedup_processing(db, dedup_key):
    db.collection("dedup").document(dedup_key).set(
        {"status": "processing", "createdAt": firestore.SERVER_TIMESTAMP}
    )


def _set_dedup_completed(db, dedup_key, response_data):
    db.collection("dedup").document(dedup_key).set(
        {"status": "completed", "response": response_data, "completedAt": firestore.SERVER_TIMESTAMP},
        merge=True,
    )


def _initial_state(*, raw_message, patient_id, conversation_id, message_id,
                    is_ambiguous=False, is_outbound=False, route="care_need"):
    """Build the initial ArulState dict for a new graph invocation."""
    return {
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
        "is_outbound":         is_outbound,
        "route":               route,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@https_fn.on_request(**_FN_OPTS, timeout_sec=120)
def message(req: https_fn.Request) -> https_fn.Response:
    """Patient sends an inbound message."""
    if req.method != "POST":
        return https_fn.Response("Method not allowed", status=405)
    try:
        body            = req.get_json(silent=True) or {}
        patient_id      = str(body.get("patientId", "")).strip()
        conversation_id = str(body.get("conversationId", "")).strip()
        raw_message     = str(body.get("message", "")).strip()
        is_ambiguous    = bool(body.get("isAmbiguous", False))

        if not all([patient_id, conversation_id, raw_message]):
            return _error_response("Missing: patientId, conversationId, message")

        db = get_db()
        dedup_key = f"msg_{conversation_id}_{_dedup_hash(raw_message)}"

        cached = _check_dedup(db, dedup_key)
        if cached:
            return _json_response(cached)

        _set_dedup_processing(db, dedup_key)

        message_id = str(uuid.uuid4())
        result = get_graph().invoke(
            _initial_state(
                raw_message=raw_message,
                patient_id=patient_id,
                conversation_id=conversation_id,
                message_id=message_id,
                is_ambiguous=is_ambiguous,
            ),
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
        return _json_response(response_data)
    except Exception:
        log.exception("request failed")
        return _error_response("Internal server error", 500)


@https_fn.on_request(**_FN_OPTS, timeout_sec=120)
def navigator_reply(req: https_fn.Request) -> https_fn.Response:
    """Navigator responds to a pending task (resolve or followup)."""
    if req.method != "POST":
        return https_fn.Response("Method not allowed", status=405)
    try:
        body               = req.get_json(silent=True) or {}
        conversation_id    = str(body.get("conversationId", "")).strip()
        navigator_response = str(body.get("navigatorResponse", "")).strip()
        action             = str(body.get("action", "resolve")).strip()
        reply_to_msg_id    = str(body.get("replyToMessageId", "")).strip() or None

        if not all([conversation_id, navigator_response]):
            return _error_response("Missing: conversationId, navigatorResponse")

        db = get_db()
        dedup_key = f"nav_{conversation_id}_{_dedup_hash(navigator_response)}"

        cached = _check_dedup(db, dedup_key)
        if cached:
            return _json_response(cached)

        _set_dedup_processing(db, dedup_key)

        result = get_graph().invoke(
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
        return _json_response(response_data)
    except Exception:
        log.exception("request failed")
        return _error_response("Internal server error", 500)


@https_fn.on_request(**_FN_OPTS, timeout_sec=120)
def patient_reply(req: https_fn.Request) -> https_fn.Response:
    """Patient replies to a navigator's followup question."""
    if req.method != "POST":
        return https_fn.Response("Method not allowed", status=405)
    try:
        body            = req.get_json(silent=True) or {}
        conversation_id = str(body.get("conversationId", "")).strip()
        raw_message     = str(body.get("message", "")).strip()
        patient_id      = str(body.get("patientId", "")).strip()

        if not all([conversation_id, raw_message, patient_id]):
            return _error_response("Missing: conversationId, message, patientId")

        db = get_db()

        task_snap = list(
            db.collection("tasks")
            .where("conversationId", "==", conversation_id)
            .where("status", "==", "awaiting_patient")
            .limit(1)
            .stream()
        )
        if not task_snap:
            return _error_response("No awaiting_patient task found for this conversation", 409)

        dedup_key = f"pat_{conversation_id}_{_dedup_hash(raw_message)}"

        cached = _check_dedup(db, dedup_key)
        if cached:
            return _json_response(cached)

        _set_dedup_processing(db, dedup_key)

        result = get_graph().invoke(
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
        return _json_response(response_data)
    except Exception:
        log.exception("request failed")
        return _error_response("Internal server error", 500)


@https_fn.on_request(
    **{**_FN_OPTS, "cors": options.CorsOptions(cors_origins=["*"], cors_methods=["GET", "POST", "OPTIONS"])},
    timeout_sec=60,
)
def onboard_patient(req: https_fn.Request) -> https_fn.Response:
    """Generate and send a welcome message for a newly enrolled patient."""
    if req.method != "POST":
        return https_fn.Response("Method not allowed", status=405)
    try:
        body           = req.get_json(silent=True) or {}
        patient_id     = str(body.get("patientId", "")).strip()
        navigator_name = str(body.get("navigatorName", "your care team")).strip()

        if not patient_id:
            return _error_response("Missing: patientId")

        db = get_db()
        doc = db.collection("patients").document(patient_id).get()
        if not doc.exists:
            return _error_response("Patient not found", 404)

        data            = doc.to_dict()
        patient_name    = data.get("preferredName") or data.get("name") or "there"
        cancer_type     = data.get("cancerType") or "cancer"
        treatment_phase = data.get("treatmentPhase") or "treatment"

        conversation_id = f"conv_onboard_{patient_id}"
        existing = db.collection("conversations").document(conversation_id).get()
        if existing.exists and existing.to_dict().get("welcomeDelivered"):
            return _error_response("Patient already onboarded", 409)

        prompt = WELCOME_PROMPT.format(
            patient_name=patient_name,
            cancer_type=cancer_type,
            treatment_phase=treatment_phase,
            navigator_name=navigator_name,
        )
        welcome_message = get_llm_warm().invoke(prompt).content.strip()
        now = firestore.SERVER_TIMESTAMP

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

        return _json_response({
            "success":        True,
            "welcomeMessage": welcome_message,
            "conversationId": conversation_id,
            "messageId":      msg_id,
        })
    except Exception:
        log.exception("request failed")
        return _error_response("Internal server error", 500)


@https_fn.on_request(**_FN_OPTS, timeout_sec=30)
def classify_message(req: https_fn.Request) -> https_fn.Response:
    """Classify whether a new message continues an existing conversation or starts a new one."""
    if req.method != "POST":
        return https_fn.Response("Method not allowed", status=405)
    try:
        body       = req.get_json(silent=True) or {}
        patient_id = str(body.get("patientId", "")).strip()
        raw_msg    = str(body.get("message", "")).strip()

        if not all([patient_id, raw_msg]):
            return _error_response("Missing: patientId, message")

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
                open_convs.append(f"- conversationId: {td['conversationId']} | issue: {td['summary']}")

        if not open_convs:
            return _json_response({"decision": "new", "conversationId": None, "confidence": "high", "reasoning": "No open conversations"})

        prompt = CLASSIFIER_PROMPT.format(
            new_message=raw_msg,
            open_conversations="\n".join(open_convs),
        )
        response = get_llm().invoke(prompt)
        raw = response.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"decision": "new", "conversationId": None, "confidence": "low", "reasoning": "Parse error"}

        if result.get("confidence") == "low":
            result["decision"] = "new"
            result["conversationId"] = None

        return _json_response(result)
    except Exception:
        log.exception("request failed")
        return _error_response("Internal server error", 500)


@https_fn.on_request(**_FN_OPTS, timeout_sec=120)
def outbound_message(req: https_fn.Request) -> https_fn.Response:
    """Navigator sends a proactive message to a patient."""
    if req.method != "POST":
        return https_fn.Response("Method not allowed", status=405)
    try:
        body                     = req.get_json(silent=True) or {}
        patient_id               = str(body.get("patientId", "")).strip()
        conversation_id          = str(body.get("conversationId", "")).strip()
        raw_message              = str(body.get("message", "")).strip()
        navigator_id             = str(body.get("navigatorId", "")).strip()
        is_existing_conversation = bool(body.get("isExistingConversation", False))

        if not all([patient_id, conversation_id, raw_message, navigator_id]):
            return _error_response("Missing: patientId, conversationId, message, navigatorId")

        db = get_db()
        message_id = str(uuid.uuid4())
        now = firestore.SERVER_TIMESTAMP

        # Existing conversations bypass the graph — just write directly to Firestore.
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

            return _json_response({
                "success":        True,
                "messageId":      message_id,
                "conversationId": conversation_id,
                "taskId":         task_id,
            })

        # New outbound conversation — run through the graph.
        result = get_graph().invoke(
            _initial_state(
                raw_message=raw_message,
                patient_id=patient_id,
                conversation_id=conversation_id,
                message_id=message_id,
                is_outbound=True,
                route="outbound",
            ),
            {"configurable": {"thread_id": conversation_id}},
        )
        return _json_response({
            "success":        True,
            "messageId":      message_id,
            "conversationId": conversation_id,
            "taskId":         result.get("task_id", ""),
        })
    except Exception:
        log.exception("request failed")
        return _error_response("Internal server error", 500)
