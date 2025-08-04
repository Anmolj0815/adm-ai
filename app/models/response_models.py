from pydantic import BaseModel, HttpUrl
from typing import Optional, List

class AdmissionDecisionResponse(BaseModel):
    Decision: str
    Amount: Optional[float] = None
    Justification: str
    ClausesUsed: List[str]
    VoiceOutputUrl: Optional[str] = None
