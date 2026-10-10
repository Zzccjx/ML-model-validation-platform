import uuid
from datetime import datetime
from pydantic import BaseModel

class ReportResponse(BaseModel):
    """Stub for Phase 15. Report generation not yet implemented."""
    run_id: uuid.UUID
    status: str = 'not_implemented'
    message: str = 'Report generation will be available in a future release.'
