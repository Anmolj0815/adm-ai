from fastapi import FastAPI
from .api.routes import router
from .services.llm_service import LLMService
import os

app = FastAPI(
    title="Swafinix AI Admission Inquiry Assistant",
    description="AI-powered assistant to answer admission queries based on pre-indexed policy documents.",
    version="1.0.0",
)

llm_service = LLMService()

@app.on_event("startup")
async def startup_event():
    """
    This function runs once when the application starts.
    It triggers the pre-indexing process for all documents from the repository.
    """
    print("Pre-indexing documents on startup...")
    llm_service.indexed_documents_data = llm_service.document_processor.pre_index_documents()
    print("Pre-indexing complete. API is ready.")

app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Swafinix AI Admission Inquiry Assistant API!"}
