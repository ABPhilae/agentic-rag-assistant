import logging
import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from src.agent.graph import agent_graph
from src.models import AgentRequest, AgentResponse, ApprovalRequest, UploadResponse
from src.config import get_settings
import tempfile
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(
    title='Agentic RAG Assistant API',
    description='LangGraph-powered audit agent with NeMo Guardrails',
    version='1.0.0'
)


@app.get('/health')
def health():
    return {'status': 'ok', 'agent': 'ready'}


@app.post('/agent/invoke', response_model=AgentResponse)
async def invoke_agent(request: AgentRequest):
    """
    Send a message to the agent and wait for the complete response.
    Use this for simple queries. For streaming, use /agent/stream.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {'configurable': {'thread_id': thread_id}}
    initial_state = {
        'messages': [HumanMessage(content=request.message)],
        'question_type': '',
        'retrieved_docs': [],
        'sources': [],
        'compliance_gaps': [],
        'deadline_warnings': [],
        'needs_approval': False,
        'final_response': '',
        'steps_taken': [],
        'thread_id': thread_id,
    }
    try:
        result = agent_graph.invoke(initial_state, config)
        return AgentResponse(
            response=result.get('final_response', 'No response generated'),
            thread_id=thread_id,
            steps_taken=result.get('steps_taken', []),
            sources=result.get('sources', []),
            requires_human_approval=result.get('needs_approval', False),
        )
    except Exception as e:
        logger.error(f'Agent invocation failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/agent/stream')
async def stream_agent(request: AgentRequest):
    """
    Stream the agent's intermediate steps in real time.
    Returns Server-Sent Events (SSE). The Streamlit frontend listens to these.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {'configurable': {'thread_id': thread_id}}
    initial_state = {
        'messages': [HumanMessage(content=request.message)],
        'question_type': '', 'retrieved_docs': [], 'sources': [],
        'compliance_gaps': [], 'deadline_warnings': [],
        'needs_approval': False, 'final_response': '',
        'steps_taken': [], 'thread_id': thread_id,
    }

    def event_generator():
        for chunk in agent_graph.stream(initial_state, config, stream_mode='updates'):
            for node_name, node_output in chunk.items():
                event = {
                    'node': node_name,
                    'steps': node_output.get('steps_taken', []),
                    'response': node_output.get('final_response', ''),
                    'needs_approval': node_output.get('needs_approval', False)
                }
                yield f'data: {json.dumps(event)}\n\n'

    return StreamingResponse(event_generator(), media_type='text/event-stream')


@app.post('/agent/approve')
async def approve_action(request: ApprovalRequest):
    """
    Submit human approval/rejection for a paused agent workflow.
    The agent resumes from where it paused (human_review node).
    """
    config = {'configurable': {'thread_id': request.thread_id}}
    try:
        result = agent_graph.invoke(
            None,        # None = resume from checkpoint
            config,
            command={'resume': request.decision}
        )
        return {
            'status': 'resumed',
            'decision': request.decision,
            'response': result.get('final_response', ''),
            'reviewer': request.reviewer_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/documents/upload', response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF and index it into Qdrant.
    Reused from Phase 3 — same indexing pipeline.
    """
    from src.services.rag_service import index_document
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail='Only PDF files are supported')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        chunks = await index_document(tmp_path, file.filename)
        return UploadResponse(
            filename=file.filename,
            chunks_indexed=chunks,
            status='indexed'
        )
    finally:
        os.unlink(tmp_path)
