from pydantic import BaseModel
from typing import Optional, List


class AgentRequest(BaseModel):
    message: str
    thread_id: str = 'default'          # Each thread = one conversation
    require_approval: bool = True        # Enable human-in-the-loop


class AgentResponse(BaseModel):
    response: str
    thread_id: str
    steps_taken: List[str] = []          # Which nodes were executed
    sources: List[str] = []              # Document sources used
    requires_human_approval: bool = False
    pending_action: Optional[str] = None


class ApprovalRequest(BaseModel):
    thread_id: str
    decision: str                        # 'approved' or 'rejected'
    reviewer_name: str = 'Auditor'


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    status: str
