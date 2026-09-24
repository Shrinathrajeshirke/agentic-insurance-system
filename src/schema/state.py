## Defines the state structure passed between LangGraph nodes

from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from src.schema.user_profile import UserProfile

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    user_profile: UserProfile
    retrieved_policies: List[Dict[str, Any]]
    underwriting_alerts: List[str]
    next_action: str