from langgraph.graph import StateGraph, START, END

from state import ArulState
from nodes import (
    intake_node,
    supervisor_node,
    ai_reply_node,
    write_task_node,
    generate_ack_node,
    format_outbound_node,
    wait_for_navigator_node,
    write_followup_node,
    wait_for_patient_node,
    format_reply_node,
)

SUPERVISOR_ROUTES = {
    "chitchat": "ai_reply",
    "care_need": "write_task",
    "outbound": "write_task",
}

POST_TASK_ROUTES = {
    "outbound": "format_outbound",
    "care_need": "generate_ack",
}

NAVIGATOR_ROUTES = {
    "resolve": "format_reply",
    "followup": "write_followup",
}


def route_from_supervisor(state: ArulState) -> str:
    route = state.get("route", "care_need")
    return SUPERVISOR_ROUTES.get(route, "write_task")


def route_after_write_task(state: ArulState) -> str:
    route = state.get("route", "care_need")
    return POST_TASK_ROUTES.get(route, "generate_ack")


def route_after_navigator(state: ArulState) -> str:
    action = state.get("navigator_action", "resolve")
    return NAVIGATOR_ROUTES.get(action, "format_reply")


def build_graph(checkpointer):
    g = StateGraph(ArulState)
    
    g.add_node("intake", intake_node)
    g.add_node("supervisor", supervisor_node)
    g.add_node("ai_reply", ai_reply_node)
    g.add_node("write_task", write_task_node)
    g.add_node("generate_ack", generate_ack_node)
    g.add_node("format_outbound", format_outbound_node)
    g.add_node("wait_for_navigator", wait_for_navigator_node)
    g.add_node("write_followup", write_followup_node)
    g.add_node("wait_for_patient", wait_for_patient_node)
    g.add_node("format_reply", format_reply_node)
    
    g.add_edge(START, "intake")
    g.add_edge("intake", "supervisor")
    
    g.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        SUPERVISOR_ROUTES
    )
    
    g.add_edge("ai_reply", END)
    
    g.add_conditional_edges(
        "write_task",
        route_after_write_task,
        POST_TASK_ROUTES
    )
    
    g.add_edge("generate_ack", "wait_for_navigator")
    g.add_edge("format_outbound", END)
    
    g.add_conditional_edges(
        "wait_for_navigator",
        route_after_navigator,
        NAVIGATOR_ROUTES
    )
    
    g.add_edge("write_followup", "wait_for_patient")
    g.add_edge("wait_for_patient", "wait_for_navigator")
    g.add_edge("format_reply", END)
    
    return g.compile(checkpointer=checkpointer)
