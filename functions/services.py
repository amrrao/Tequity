import os
import firebase_admin
from firebase_admin import firestore
from langgraph_checkpoint_firestore import FirestoreSaver

_db = None
_llm = None
_llm_warm = None
_checkpointer = None
_graph = None


def _ensure_firebase():
    if not firebase_admin._apps:
        firebase_admin.initialize_app()


def get_db():
    global _db
    if _db is None:
        _ensure_firebase()
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
    global _llm_warm
    if _llm_warm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        _llm_warm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            temperature=0.5,
        )
    return _llm_warm


def get_checkpointer():
    global _checkpointer
    if _checkpointer is None:
        _ensure_firebase()
        _checkpointer = FirestoreSaver(
            project_id=os.environ.get("GCLOUD_PROJECT"),
            checkpoints_collection="lg_checkpoints",
            writes_collection="lg_writes",
        )
    return _checkpointer


def get_graph():
    global _graph
    if _graph is not None:
        return _graph
    from graph import build_graph
    _graph = build_graph(get_checkpointer())
    return _graph
