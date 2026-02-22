from typing import TypedDict, Annotated, Optional, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Shared state that flows through the entire LangGraph.
    Think of this as the agent's shared notepad.
    """
    # The conversation so far (Annotated with add_messages = append, not replace)
    messages: Annotated[List[BaseMessage], add_messages]

    # What the agent decided the question is about
    question_type: str            # 'simple' or 'complex'

    # Documents retrieved from Qdrant
    retrieved_docs: List[dict]

    # Source filenames used
    sources: List[str]

    # Results from compliance check tool
    compliance_gaps: List[str]

    # Results from deadline check tool
    deadline_warnings: List[str]

    # Whether this requires human approval
    needs_approval: bool

    # The final response text
    final_response: str

    # Execution trace for the UI
    steps_taken: List[str]

    # Thread config
    thread_id: str
