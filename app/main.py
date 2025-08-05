from fastapi import FastAPI
from .api.routes import router
from .services.llm_service import LLMService
import os

app = FastAPI(
    title="Swafinix AI Admission Inquiry Assistant",
    description="AI-powered assistant to answer admission queries based on pre-indexed policy documents.",
    version="1.0.0",
)

app.include_router(router)

@app.on_event("startup")
async def startup_event():
    """
    This function runs once when the application starts.
    It triggers the pre-indexing process for all documents.
    """
    llm_service = LLMService()
    # Check if embeddings file exists
    if not os.path.exists("data/embeddings/admissions_embeddings.pkl"):
        print("Embeddings file not found. Pre-indexing documents...")
        llm_service.document_processor.pre_index_documents()
    else:
        print("Embeddings file found. Skipping pre-indexing on startup.")

@app.get("/")
async def root():
    return {"message": "Welcome to the Swafinix AI Admission Inquiry Assistant API!"}
