import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.types import interrupt
from src.agent.state import AgentState
from src.agent.tools import (
    search_audit_documents, check_compliance_gaps,
    check_remediation_deadlines, generate_executive_summary
)
from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def get_llm():
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        openai_api_key=settings.openai_api_key
    )


def classify_question(state: AgentState) -> dict:
    """
    NODE 1: Classify the user's message as 'simple' or 'complex'.
    Simple = a direct question about a document (fast RAG path).
    Complex = a multi-step task requiring analysis, comparison, or a report.
    """
    user_message = state['messages'][-1].content
    llm = get_llm()
    prompt = f"""Classify this user message as either 'simple' or 'complex'.
    Simple: a direct question that needs one search (e.g., 'What is finding HK-001?')
    Complex: a task requiring multiple steps (e.g., 'Review all critical findings
    and identify compliance gaps, then prepare a report')
    User message: {user_message}
    Answer with only one word: simple or complex"""
    response = llm.invoke([HumanMessage(content=prompt)])
    q_type = response.content.strip().lower()
    if q_type not in ['simple', 'complex']:
        q_type = 'simple'
    logger.info(f'Question classified as: {q_type}')
    return {
        'question_type': q_type,
        'steps_taken': state.get('steps_taken', []) + [f'Classified as: {q_type}']
    }


def fast_rag(state: AgentState) -> dict:
    """
    NODE 2 (simple path): Direct RAG retrieval — same as Phase 3 but faster.
    Skips multi-step planning and goes straight to document search.
    """
    query = state['messages'][-1].content
    search_result = search_audit_documents.invoke({'query': query, 'top_k': 5})
    docs = [{'content': search_result, 'source': 'qdrant_search'}]
    sources = ['audit_documents']
    return {
        'retrieved_docs': docs,
        'sources': sources,
        'steps_taken': state.get('steps_taken', []) + ['Fast RAG retrieval']
    }


def plan_steps(state: AgentState) -> dict:
    """
    NODE 3 (complex path): Decide which tools to invoke.
    For complex tasks, we run all analysis tools in parallel.
    """
    return {
        'needs_approval': True,   # Complex tasks always need approval
        'steps_taken': state.get('steps_taken', []) + ['Planning multi-step analysis']
    }


def search_docs(state: AgentState) -> dict:
    """NODE 4: Run the document search tool."""
    query = state['messages'][-1].content
    result = search_audit_documents.invoke({'query': query, 'top_k': 8})
    return {
        'retrieved_docs': [{'content': result, 'source': 'qdrant'}],
        'steps_taken': state.get('steps_taken', []) + ['Searched audit documents']
    }


def check_compliance(state: AgentState) -> dict:
    """NODE 5: Run the compliance gap check tool."""
    docs_summary = ' '.join([d.get('content', '')[:400]
                              for d in state.get('retrieved_docs', [])])
    if not docs_summary.strip():
        docs_summary = state['messages'][-1].content
    result = check_compliance_gaps.invoke({'finding_summary': docs_summary})
    has_gaps = len(result) > 20 and 'no gap' not in result.lower()
    return {
        'compliance_gaps': [result],
        'needs_approval': has_gaps,
        'steps_taken': state.get('steps_taken', []) + ['Compliance gap check complete']
    }


def check_deadlines(state: AgentState) -> dict:
    """NODE 6: Check upcoming remediation deadlines."""
    result = check_remediation_deadlines.invoke({'days_threshold': 30})
    return {
        'deadline_warnings': [result],
        'steps_taken': state.get('steps_taken', []) + ['Deadline check complete']
    }


def human_review_node(state: AgentState) -> dict:
    """
    NODE 7: PAUSE and wait for human approval.
    In LangGraph, interrupt() pauses execution and saves state.
    The workflow resumes when the /agent/approve endpoint is called.
    This is the banking compliance control gate.
    """
    gaps = state.get('compliance_gaps', ['No gaps identified'])
    warnings = state.get('deadline_warnings', [])
    message = f"""HUMAN APPROVAL REQUIRED
    Compliance gaps found: {len(gaps)} issue(s)
    Deadline warnings: {len(warnings)} item(s)
    Please review and approve or reject report generation."""
    decision = interrupt(message)   # <-- PAUSES HERE
    if decision == 'approved':
        return {
            'steps_taken': state.get('steps_taken', []) + ['Human approval granted'],
            'needs_approval': False
        }
    return {
        'final_response': 'Report generation rejected by reviewer.',
        'steps_taken': state.get('steps_taken', []) + ['Human approval rejected']
    }


def generate_response(state: AgentState) -> dict:
    """
    NODE 8 (final): Generate the response for the user.
    For simple questions: synthesise the RAG results into a clear answer.
    For complex tasks: generate a full executive summary report.
    """
    if state.get('final_response'):   # If rejected by human
        return {}
    llm = get_llm()
    question_type = state.get('question_type', 'simple')
    user_query = state['messages'][-1].content
    docs = state.get('retrieved_docs', [])
    gaps = state.get('compliance_gaps', [])
    warnings = state.get('deadline_warnings', [])
    if question_type == 'simple':
        context = '\n'.join([d.get('content', '')[:1000] for d in docs])
        prompt = f"""Answer this question using only the provided context.
        Question: {user_query}
        Context: {context}
        If the answer is not in the context, say so clearly."""
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content
    else:
        findings = '\n'.join([d.get('content', '')[:500] for d in docs])
        gaps_text = '\n'.join(gaps) if gaps else 'None identified'
        answer = generate_executive_summary.invoke({
            'findings': findings,
            'compliance_gaps': gaps_text
        })
        if warnings:
            answer += '\n\n---\nDEADLINE ALERTS:\n' + '\n'.join(warnings)
    return {
        'final_response': answer,
        'steps_taken': state.get('steps_taken', []) + ['Response generated']
    }
