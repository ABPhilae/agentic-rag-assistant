from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter
from src.config import get_settings
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
settings = get_settings()


@tool
def search_audit_documents(query: str, top_k: int = 5) -> str:
    """
    Search the audit document database for findings, policies, or procedures.
    Use this when the user asks about specific audit findings, control gaps,
    remediation status, or any document content.
    Returns relevant document excerpts with their source filenames.
    """
    try:
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key
        )
        query_vector = embeddings.embed_query(query)
        results = client.search(
            collection_name=settings.qdrant_collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )
        if not results:
            return 'No relevant documents found in the audit database.'
        output = []
        for i, hit in enumerate(results, 1):
            source = hit.payload.get('source', 'Unknown')
            content = hit.payload.get('page_content', '')[:800]
            score = round(hit.score, 3)
            output.append(f'[{i}] Source: {source} (relevance: {score})')
            output.append(f'    {content}')
        return '\n'.join(output)
    except Exception as e:
        logger.error(f'search_audit_documents failed: {e}')
        return f'Document search failed: {str(e)}'


@tool
def check_compliance_gaps(finding_summary: str) -> str:
    """
    Check a set of audit findings for common compliance gaps.
    Use this when verifying whether findings meet regulatory requirements
    (HKMA, MAS, FATF). Checks for: missing remediation owner, missing
    deadline, missing budget allocation, and overdue status.
    Input: a summary of findings. Returns: identified gaps.
    """
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=settings.openai_model, temperature=0)
    prompt = f"""You are a compliance officer reviewing audit findings.
    Analyse these findings for regulatory compliance gaps:
    {finding_summary}

    Check each finding for these REQUIRED attributes:
    1. Remediation owner (named individual, not just a department)
    2. Target completion date (specific date)
    3. Budget allocated (amount or 'within existing budget')
    4. Current status (Open/In Progress/Closed)

    List any missing attributes as GAPS. Reference HKMA or MAS guidelines
    where applicable. Be specific and concise."""
    response = llm.invoke(prompt)
    return response.content


@tool
def check_remediation_deadlines(days_threshold: int = 30) -> str:
    """
    Check for audit findings with remediation deadlines within the specified
    number of days (default: 30). Use this when the user asks about upcoming
    deadlines, overdue items, or time-sensitive findings.
    Returns a list of at-risk findings with their owners and deadlines.
    """
    # In a real system, this queries a database.
    # For the portfolio project, we simulate with realistic sample data.
    today = datetime.now()
    sample_findings = [
        {'id': 'HK-2024-001', 'title': 'Trade reconciliation control gap',
         'owner': 'Alice Chen', 'deadline': '2026-03-15', 'status': 'In Progress'},
        {'id': 'HK-2024-007', 'title': 'AML transaction monitoring threshold',
         'owner': 'Bob Lam', 'deadline': '2026-02-28', 'status': 'Open'},
        {'id': 'SG-2024-003', 'title': 'Access control review — trading system',
         'owner': 'Carol Tan', 'deadline': '2026-04-30', 'status': 'In Progress'},
    ]
    at_risk = []
    for f in sample_findings:
        deadline = datetime.strptime(f['deadline'], '%Y-%m-%d')
        days_remaining = (deadline - today).days
        if days_remaining <= days_threshold:
            at_risk.append(
                f"Finding {f['id']}: '{f['title']}' | Owner: {f['owner']} | "
                f"Deadline: {f['deadline']} ({days_remaining} days) | Status: {f['status']}"
            )
    if not at_risk:
        return f'No findings with deadlines within {days_threshold} days.'
    return 'AT-RISK FINDINGS:\n' + '\n'.join(at_risk)


@tool
def generate_executive_summary(findings: str, compliance_gaps: str) -> str:
    """
    Generate a professional executive summary report from audit findings
    and compliance gap analysis. Use this as the FINAL STEP after all
    analysis is complete and human approval has been granted.
    Returns a formatted executive summary suitable for senior management.
    """
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=settings.openai_model, temperature=0.2)
    prompt = f"""You are a senior internal auditor preparing an executive summary.
    Based on the following findings and compliance analysis, write a concise
    executive summary suitable for the Chief Audit Executive.

    FINDINGS:\n{findings}

    COMPLIANCE GAPS:\n{compliance_gaps}

    Format:
    - Executive Summary (2-3 sentences)
    - Key Findings (numbered list, max 5)
    - Compliance Gaps (numbered list)
    - Recommended Actions (numbered list, prioritised by risk)
    - Conclusion
    Keep the total under 400 words."""
    response = llm.invoke(prompt)
    return response.content


# Export all tools as a list for the agent to use
ALL_TOOLS = [
    search_audit_documents,
    check_compliance_gaps,
    check_remediation_deadlines,
    generate_executive_summary,
]
