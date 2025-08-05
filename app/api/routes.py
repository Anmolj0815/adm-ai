from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from ..services.llm_service import LLMService
from ..services.voice_service import VoiceService
from ..models.response_models import AdmissionDecisionResponse

router = APIRouter()
llm_service = LLMService()
voice_service = VoiceService()

class QueryRequest(BaseModel):
    query: str
    document_urls: List[str] = []  # Changed HttpUrl to str
    voice_input: bool = False
    return_voice: bool = False

@router.post("/inquire", response_model=AdmissionDecisionResponse)
async def inquire_admission(request: QueryRequest):
    try:
        if request.voice_input:
            text_query = await voice_service.speech_to_text(request.query)
        else:
            text_query = request.query

        decision_response = await llm_service.process_admission_query(
            text_query,
            document_urls=request.document_urls
        )

        if request.return_voice:
            voice_output_url = await voice_service.text_to_speech(decision_response.Justification)
            decision_response.VoiceOutputUrl = voice_output_url

        return decision_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
