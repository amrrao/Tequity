from typing import TypedDict, Optional, Literal

TaskType = Literal[
    "appointment",
    "medication",
    "insurance",
    "transportation",
    "meal",
    "emotional_support",
    "general",
    "unclear",
    "chitchat",
]

Urgency = Literal["urgent", "routine", "low"]

Route = Literal["chitchat", "care_need", "outbound"]


class ArulState(TypedDict):
    raw_message: str
    patient_id: str
    conversation_id: str
    patient_name: str
    patient_context: dict
    task_id: str
    task_type: TaskType
    urgency: Urgency
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
    is_outbound: bool
    route: Route
