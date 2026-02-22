from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.agent.state import AgentState
from src.agent import nodes


def route_after_classify(state: AgentState) -> str:
    """
    Conditional edge: after classifying the question,
    route to fast_rag (simple) or plan_steps (complex).
    This is the first fork in the flowchart.
    """
    if state['question_type'] == 'simple':
        return 'fast_rag'
    return 'plan_steps'


def route_after_planning(state: AgentState) -> str:
    """
    After planning, check if we need human approval.
    If the compliance check found gaps, require approval before reporting.
    """
    if state.get('needs_approval', False):
        return 'human_review'
    return 'generate_response'


def build_agent_graph():
    """Build and compile the LangGraph agent."""
    # Create the graph builder with our state type
    builder = StateGraph(AgentState)

    # Register all nodes (functions from nodes.py)
    builder.add_node('classify_question',  nodes.classify_question)
    builder.add_node('fast_rag',            nodes.fast_rag)
    builder.add_node('plan_steps',          nodes.plan_steps)
    builder.add_node('search_docs',         nodes.search_docs)
    builder.add_node('check_compliance',    nodes.check_compliance)
    builder.add_node('check_deadlines',     nodes.check_deadlines)
    builder.add_node('human_review',        nodes.human_review_node)
    builder.add_node('generate_response',   nodes.generate_response)

    # Wire the graph (the arrows in your flowchart)
    builder.add_edge(START, 'classify_question')

    # Conditional fork after classification
    builder.add_conditional_edges('classify_question', route_after_classify,
        {'fast_rag': 'fast_rag', 'plan_steps': 'plan_steps'})

    # Simple path: fast_rag -> generate_response -> END
    builder.add_edge('fast_rag', 'generate_response')

    # Complex path: plan_steps -> parallel tools
    builder.add_edge('plan_steps', 'search_docs')
    builder.add_edge('plan_steps', 'check_compliance')
    builder.add_edge('plan_steps', 'check_deadlines')

    # All tools converge back to approval check
    builder.add_edge('search_docs',      'generate_response')
    builder.add_edge('check_compliance', 'generate_response')
    builder.add_edge('check_deadlines',  'generate_response')

    # Conditional fork before generating response
    builder.add_conditional_edges('check_compliance', route_after_planning,
        {'human_review': 'human_review', 'generate_response': 'generate_response'})

    builder.add_edge('human_review',     'generate_response')
    builder.add_edge('generate_response', END)

    # Compile with in-memory checkpointer
    # In production (Step 10 Docker), we swap this for RedisCheckpointer
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# Build once at import time — reused by FastAPI endpoints
agent_graph = build_agent_graph()
