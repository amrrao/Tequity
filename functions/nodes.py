import json
import uuid
from firebase_admin import firestore
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import interrupt

from state import ArulState
from prompts import SUPERVISOR_PROMPT, CHITCHAT_PROMPT, ACK_PROMPT, URGENT_ACK
from services import get_db, get_llm, get_llm_warm


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
    if state.get("is_outbound"):
        return {
            "route": "outbound",
            "task_id": str(uuid.uuid4()),
            "task_type": "general",
            "urgency": "routine",
            "summary": f"Navigator-initiated: {state['raw_message'][:80]}",
            "navigator_notes": "",
            "extracted_entities": {"dates": [], "locations": [], "people": [], "organizations": [], "other": []},
        }
    
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
    
    task_type = parsed.get("taskType", "unclear")
    route = "chitchat" if task_type == "chitchat" else "care_need"
    
    return {
        "route":              route,
        "task_id":            str(uuid.uuid4()),
        "task_type":          task_type,
        "urgency":            parsed.get("urgency", "routine"),
        "summary":            parsed.get("summary", ""),
        "navigator_notes":    parsed.get("navigatorNotes", ""),
        "extracted_entities": parsed.get("extractedEntities", {}),
    }


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
    is_outbound = state.get("is_outbound", False)

    conv_doc = db.collection("conversations").document(state["conversation_id"]).get()
    is_first_message = not conv_doc.exists

    batch = db.batch()

    if not is_outbound:
        msg_ref = db.collection("messages").document(msg_id)
        batch.set(msg_ref, {
            "messageId":        msg_id,
            "conversationId":   state["conversation_id"],
            "patientId":        state["patient_id"],
            "sender":           "patient",
            "body":             state["raw_message"],
            "replyToMessageId": state.get("reply_to_message_id") or None,
            "taskId":           state["task_id"],
            "createdAt":        now,
        })

    task_ref = db.collection("tasks").document(state["task_id"])
    batch.set(task_ref, {
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

    conv_ref = db.collection("conversations").document(state["conversation_id"])
    batch.set(conv_ref, {
        "lastActivity": now,
        "lastMessage":  state["raw_message"],
        "status":       "awaiting_navigator" if not is_outbound else "awaiting_patient",
        "patientId":    state["patient_id"],
    }, merge=True)

    batch.commit()

    return {}


def generate_ack_node(state: ArulState) -> dict:
    db = get_db()
    conv_doc = db.collection("conversations").document(state["conversation_id"]).get()
    is_first_message = not conv_doc.exists or not conv_doc.to_dict().get("lastActivity")
    
    if not is_first_message:
        return {"ack_message": ""}
    
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


def format_outbound_node(state: ArulState) -> dict:
    db = get_db()
    now = firestore.SERVER_TIMESTAMP
    msg_id = state["message_id"]

    batch = db.batch()

    msg_ref = db.collection("messages").document(msg_id)
    batch.set(msg_ref, {
        "messageId":        msg_id,
        "conversationId":   state["conversation_id"],
        "patientId":        state["patient_id"],
        "sender":           "navigator",
        "body":             state["raw_message"],
        "replyToMessageId": None,
        "taskId":           state["task_id"],
        "createdAt":        now,
    })

    task_ref = db.collection("tasks").document(state["task_id"])
    batch.set(task_ref, {
        "status":            "completed",
        "navigatorResponse": state["raw_message"],
        "replyMessageId":    msg_id,
        "imessageDelivered": False,
        "isOutbound":        True,
        "updatedAt":         now,
    }, merge=True)

    batch.commit()

    return {"final_message": state["raw_message"]}


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


def write_followup_node(state: ArulState) -> dict:
    db = get_db()
    now = firestore.SERVER_TIMESTAMP
    question = (state.get("navigator_response") or "").strip()
    followup_msg_id = str(uuid.uuid4())

    batch = db.batch()

    msg_ref = db.collection("messages").document(followup_msg_id)
    batch.set(msg_ref, {
        "messageId":        followup_msg_id,
        "conversationId":   state["conversation_id"],
        "patientId":        state["patient_id"],
        "sender":           "navigator",
        "body":             question,
        "replyToMessageId": state["message_id"],
        "taskId":           state["task_id"],
        "createdAt":        now,
    })

    task_ref = db.collection("tasks").document(state["task_id"])
    batch.set(task_ref, {
        "status":            "awaiting_patient",
        "navigatorResponse": question,
        "replyMessageId":    followup_msg_id,
        "imessageDelivered": False,
        "updatedAt":         now,
    }, merge=True)

    conv_ref = db.collection("conversations").document(state["conversation_id"])
    batch.set(conv_ref, {"status": "awaiting_patient", "lastActivity": now}, merge=True)

    batch.commit()

    return {
        "navigator_response": question,
        "message_id":         followup_msg_id,
    }


def wait_for_patient_node(state: ArulState) -> dict:
    db = get_db()
    followup_msg_id = state["message_id"]
    question = (state.get("navigator_response") or "").strip()

    patient_reply = interrupt({
        "waiting_for":       "patient_reply",
        "taskId":            state["task_id"],
        "followupMessageId": followup_msg_id,
        "question":          question,
    })

    new_msg_id = str(uuid.uuid4())
    now = firestore.SERVER_TIMESTAMP

    batch = db.batch()

    new_msg_ref = db.collection("messages").document(new_msg_id)
    batch.set(new_msg_ref, {
        "messageId":        new_msg_id,
        "conversationId":   state["conversation_id"],
        "patientId":        state["patient_id"],
        "sender":           "patient",
        "body":             patient_reply,
        "replyToMessageId": followup_msg_id,
        "taskId":           state["task_id"],
        "createdAt":        now,
    })

    task_ref = db.collection("tasks").document(state["task_id"])
    batch.set(task_ref, {
        "status":       "pending",
        "rawMessage":   patient_reply,
        "patientReply": patient_reply,
        "updatedAt":    now,
    }, merge=True)

    conv_ref = db.collection("conversations").document(state["conversation_id"])
    batch.set(conv_ref, {
        "status":       "awaiting_navigator",
        "lastActivity": now,
        "lastMessage":  patient_reply,
    }, merge=True)

    batch.commit()

    return {
        "raw_message":        patient_reply,
        "message_id":         new_msg_id,
        "navigator_response": None,
        "navigator_action":   None,
        "summary":            "",
        "navigator_notes":    "",
    }


def format_reply_node(state: ArulState) -> dict:
    db = get_db()
    reply = (state.get("navigator_response") or "").strip() or "Your care team will follow up shortly."
    now = firestore.SERVER_TIMESTAMP

    reply_msg_id = str(uuid.uuid4())

    batch = db.batch()

    msg_ref = db.collection("messages").document(reply_msg_id)
    batch.set(msg_ref, {
        "messageId":        reply_msg_id,
        "conversationId":   state["conversation_id"],
        "patientId":        state["patient_id"],
        "sender":           "navigator",
        "body":             reply,
        "replyToMessageId": state["message_id"],
        "taskId":           state["task_id"],
        "createdAt":        now,
    })

    task_ref = db.collection("tasks").document(state["task_id"])
    batch.set(task_ref, {
        "status":            "completed",
        "navigatorResponse": reply,
        "replyMessageId":    reply_msg_id,
        "imessageDelivered": False,
        "updatedAt":         now,
    }, merge=True)

    conv_ref = db.collection("conversations").document(state["conversation_id"])
    batch.set(conv_ref, {"status": "resolved", "lastActivity": now}, merge=True)

    batch.commit()

    return {"final_message": reply, "reply_to_message_id": state["message_id"]}
