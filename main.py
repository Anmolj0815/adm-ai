import os
import requests
import asyncio
import itertools
import logging
import json
import random
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, status, Request
from pydantic import BaseModel
import pypdf
import numpy as np
import faiss
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep,
)

from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat import ChatMessage

# --- Environment Variables and Configuration ---
load_dotenv()

MISTRAL_API_KEYS_STR = os.getenv("MISTRAL_API_KEYS")
if not MISTRAL_API_KEYS_STR:
    raise ValueError("MISTRAL_API_KEYS environment variable is not set.")

MISTRAL_API_KEYS = [k.strip() for k in MISTRAL_API_KEYS_STR.split(',') if k.strip()]
if not MISTRAL_API_KEYS:
    raise ValueError("MISTRAL_API_KEYS environment variable is set but contains no valid keys.")

MISTRAL_API_KEY = MISTRAL_API_KEYS[0]

# PDF Path
PDF_DIR = "data/admission_policies"

app = FastAPI(
    title="Swafinix AI Admission Inquiry Assistant",
    description="A simple API to answer admission queries based on pre-indexed PDFs.",
    version="1.0.0",
)

# --- Global State for the RAG System ---
class RAGState:
    vector_store: Optional[faiss.IndexFlatL2] = None
    indexed_data: List[Dict[str, Any]] = []

rag_state = RAGState()

# --- Logging Configuration ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- Document Processing Functions ---
def extract_text_from_pdf(file_path: str) -> str:
    logger.info(f"Extracting text from PDF: {file_path}")
    reader = pypdf.PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def get_text_chunks(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = para
        else:
            current_chunk += ' ' + para
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(httpx.HTTPStatusError),
    before_sleep=lambda retry_state: logger.warning(f"Retrying embedding call, attempt {retry_state.attempt_number}...")
)
def embed_texts_in_batches(texts: List[str], batch_size: int = 250) -> np.ndarray:
    logger.info(f"Embedding {len(texts)} chunks in batches of {batch_size}...")
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            client = MistralClient(api_key=MISTRAL_API_KEY)
            embeddings_response = client.embeddings(
                model="mistral-embed",
                input=batch
            )
            embeddings = [data.embedding for data in embeddings_response.data]
            all_embeddings.extend(embeddings)
            logger.info(f"Processed batch {i // batch_size + 1}")
        except Exception as e:
            logger.error(f"Error embedding batch with Mistral: {e}")
            raise
    return np.array(all_embeddings).astype('float32')

# --- RAG & LLM Logic ---
def semantically_retrieve_information(query: str) -> List[str]:
    indexed_documents_data = rag_state.indexed_data
    if not indexed_documents_data:
        return ["No documents provided or processed for retrieval."]

    client = MistralClient(api_key=MISTRAL_API_KEY)
    query_embedding_response = client.embeddings(
        model="mistral-embed",
        input=[query]
    )
    query_embedding = np.array(query_embedding_response.data[0].embedding)
    
    document_embeddings = np.array([d["embedding"] for d in indexed_documents_data])
    document_texts = [d["text"] for d in indexed_documents_data]
    document_embeddings = document_embeddings.astype('float32')
    query_embedding = query_embedding.astype('float32')

    if rag_state.vector_store is None:
        dimension = document_embeddings.shape[1]
        rag_state.vector_store = faiss.IndexFlatL2(dimension)
        rag_state.vector_store.add(document_embeddings)
    
    k = min(5, len(indexed_documents_data))
    distances, indices = rag_state.vector_store.search(np.expand_dims(query_embedding, axis=0), k)

    relevant_clauses = []
    for i in indices[0]:
        if i != -1:
            relevant_clauses.append(document_texts[i])
    
    if not relevant_clauses:
        return ["No highly relevant information found in the provided documents."]
    logger.info(f"Retrieved {len(relevant_clauses)} relevant clauses using FAISS.")
    return relevant_clauses

def evaluate_with_llm(query: str, relevant_information: List[str]) -> str:
    combined_context = "\n".join(relevant_information)
    prompt = f"""
    You are an expert in analyzing admission policy documents. Your task is to answer user queries accurately and concisely, based **only** on the provided context. If the exact answer or sufficient information is not found in the context, state: "I cannot answer this question based on the provided documents." Do not generate information that is not supported by the context.

    CRITICAL INSTRUCTIONS:
    - Answer in EXACTLY 2-3 lines maximum.
    - Include specific numbers, percentages, and timeframes if relevant.
    - Start directly with the answer - no introductory phrases.

    Context:
    {combined_context}

    Question: {query}
    Answer:
    """
    messages = [
        ChatMessage(role="system", content="You are a helpful and precise admissions officer for IIM Mumbai."),
        ChatMessage(role="user", content=prompt)
    ]
    try:
        client = MistralClient(api_key=MISTRAL_API_KEY)
        chat_response = client.chat(
            model="mistral-large-latest",
            messages=messages
        )
        answer = chat_response.choices[0].message.content
        return answer
    except Exception as e:
        logger.error(f"Error evaluating with Mistral: {e}")
        return "An error occurred during decision evaluation."

def trigger_n8n_workflow(payload: Dict[str, Any]):
    """
    Makes an asynchronous POST request to the n8n webhook.
    """
    if 'N8N_WEBHOOK_URL' not in os.environ:
        logger.warning("N8N_WEBHOOK_URL not found. Skipping webhook call.")
        return

    try:
        # Use aiohttp for asynchronous requests
        async def make_request():
            async with httpx.AsyncClient() as client:
                response = await client.post(os.environ['N8N_WEBHOOK_URL'], json=payload, timeout=5)
                response.raise_for_status()
                logger.info(f"Successfully triggered n8n workflow. Response status: {response.status}")
        
        asyncio.create_task(make_request())
    except Exception as e:
        logger.error(f"Error calling n8n webhook: {e}")

# --- Pydantic Models for API Request/Response ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

# --- Application Startup Event ---
@app.on_event("startup")
async def startup_event():
    logger.info("--- Application Startup: Initializing RAG System ---")

    if rag_state.vector_store is not None:
        logger.info("RAG system already initialized. Skipping.")
        return
        
    if not os.path.exists(PDF_DIR):
        logger.error(f"ERROR: Directory '{PDF_DIR}' not found. The API cannot function without this document.")
        raise RuntimeError(f"Required directory '{PDF_DIR}' not found. Cannot start RAG service.")
    
    try:
        documents = []
        for file_name in os.listdir(PDF_DIR):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(PDF_DIR, file_name)
                logger.info(f"Loading document from: {file_path}")
                full_text = extract_text_from_pdf(file_path)
                chunks = get_text_chunks(full_text)
                
                for i, chunk in enumerate(chunks):
                    documents.append({
                        "page_content": chunk,
                        "metadata": {"source": f"{file_name}_chunk_{i}"}
                    })

        if not documents:
            logger.error("No documents found or no text extracted. RAG system will not function.")
            return

        logger.info("Splitting documents into chunks...")
        docs_for_embedding = [doc['page_content'] for doc in documents]
        
        embeddings_array = embed_texts_in_batches(docs_for_embedding)
        
        indexed_data = []
        for i, chunk in enumerate(docs_for_embedding):
            indexed_data.append({
                "embedding": embeddings_array[i],
                "text": chunk,
                "source": documents[i]['metadata']['source']
            })

        rag_state.indexed_data = indexed_data
        
        # This part of the code now creates the FAISS index on startup
        embeddings = np.array([d["embedding"] for d in rag_state.indexed_data]).astype('float32')
        dimension = embeddings.shape[1]
        rag_state.vector_store = faiss.IndexFlatL2(dimension)
        rag_state.vector_store.add(embeddings)

        logger.info("FAISS vector store built successfully. API is ready to receive requests.")

    except Exception as e:
        logger.exception(f"--- ERROR during RAG System Initialization: {e} ---")
        raise RuntimeError(f"RAG system initialization failed: {e}")

# --- API Endpoint ---
@app.post(
    "/inquire",
    response_model=QueryResponse,
    summary="Answer questions based on pre-indexed documents."
)
async def inquire_admission(request: QueryRequest):
    if rag_state.vector_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system is not initialized. Check startup logs for errors."
        )

    logger.info(f"Processing question: '{request.query}'")
    
    try:
        relevant_info = semantically_retrieve_information(request.query)
        answer = evaluate_with_llm(request.query, relevant_info)
        
        # Trigger the n8n webhook for every query
        webhook_payload = {
            "original_query": request.query,
            "answer": answer,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        trigger_n8n_workflow(webhook_payload)

        logger.info("--- Question processed. Sending response. ---")
        return {"answer": answer}

    except Exception as e:
        logger.exception(f"An unexpected error occurred while processing question '{request.query}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
