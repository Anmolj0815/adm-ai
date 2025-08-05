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

# --- Import from LangChain and MistralAI ---
from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    FAISS = None
    print("WARNING: FAISS package not found. Please install 'faiss-cpu' or 'faiss-gpu' to enable vector store functionality.")

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Environment Variables and Configuration ---
from dotenv import load_dotenv
load_dotenv()

MISTRAL_API_KEYS_STR = os.getenv("MISTRAL_API_KEYS")
if not MISTRAL_API_KEYS_STR:
    raise ValueError("MISTRAL_API_KEYS environment variable is not set.")

MISTRAL_API_KEYS = [k.strip() for k in MISTRAL_API_KEYS_STR.split(',') if k.strip()]
if not MISTRAL_API_KEYS:
    raise ValueError("MISTRAL_API_KEYS environment variable is set but contains no valid keys.")

# Use a single API key for simplicity in a non-enterprise setting
MISTRAL_API_KEY = MISTRAL_API_KEYS[0]

# --- PDF Path and FastAPI App Setup ---
PDF_DIR = "data/admission_policies"
app = FastAPI(
    title="Swafinix AI Admission Inquiry Assistant",
    description="A simple API to answer admission queries based on pre-indexed PDFs.",
    version="1.0.0",
)

# --- Global State for the RAG System ---
class RAGState:
    vector_store: Optional[FAISS] = None
    
rag_state = RAGState()

# --- Logging Configuration ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- RAG & LLM Logic ---
PROMPT_TEMPLATE = """
You are an expert in analyzing admission policy documents. Your task is to answer user queries accurately and concisely, based **only** on the provided context. If the exact answer or sufficient information is not found in the context, state: "I cannot answer this question based on the provided documents." Do not generate information that is not supported by the context.

CRITICAL INSTRUCTIONS:
- Answer in EXACTLY 2-3 lines maximum.
- Include specific numbers, percentages, and timeframes if relevant.
- Start directly with the answer - no introductory phrases.

Context:
{context}

Question: {question}
Answer:
"""
CUSTOM_PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(httpx.HTTPStatusError),
    before_sleep=lambda retry_state: logger.warning(f"Retrying embedding call, attempt {retry_state.attempt_number}...")
)
def embed_documents_with_retries(docs: List[Any]) -> FAISS:
    """Embed documents with retries."""
    embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=MISTRAL_API_KEY)
    logger.info("Creating embeddings and building FAISS vector store...")
    return FAISS.from_documents(docs, embeddings)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(httpx.HTTPStatusError),
    before_sleep=lambda retry_state: logger.warning(f"Retrying chain invocation, attempt {retry_state.attempt_number}...")
)
async def process_question_with_retries(question: str, vector_store: FAISS) -> str:
    """
    Handles the RAG chain invocation with built-in retries.
    """
    llm = ChatMistralAI(model="mistral-large-latest", temperature=0, mistral_api_key=MISTRAL_API_KEY)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": CUSTOM_PROMPT}
    )
    
    logger.info(f"Invoking RAG chain for question: '{question}'")
    
    result = await qa_chain.ainvoke({"query": question})
    
    return result.get("result", "I cannot answer this question based on the provided documents.")

# --- Application Startup Event ---
@app.on_event("startup")
async def startup_event():
    logger.info("--- Application Startup: Initializing RAG System ---")

    if FAISS is None:
        logger.error("ERROR: FAISS is not installed. Default RAG system cannot be initialized.")
        raise RuntimeError("FAISS library not installed. Cannot start RAG service.")

    if not os.path.exists(PDF_DIR):
        logger.error(f"ERROR: Directory '{PDF_DIR}' not found. The API cannot function without this document.")
        raise RuntimeError(f"Required directory '{PDF_DIR}' not found. Cannot start RAG service.")

    try:
        documents = []
        for file_name in os.listdir(PDF_DIR):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(PDF_DIR, file_name)
                logger.info(f"Loading document from: {file_path}")
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
        
        logger.info(f"Loaded {len(documents)} pages from documents in {PDF_DIR}")

        logger.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        logger.info(f"Created {len(docs)} text chunks.")

        rag_state.vector_store = embed_documents_with_retries(docs)
        logger.info("FAISS vector store built successfully. API is ready to receive requests.")

    except Exception as e:
        logger.exception(f"--- ERROR during RAG System Initialization: {e} ---")
        raise RuntimeError(f"RAG system initialization failed: {e}")

# --- API Endpoint ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

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
        answer = await process_question_with_retries(request.query, rag_state.vector_store)
        logger.info("--- Question processed. Sending response. ---")
        return {"answer": answer}

    except Exception as e:
        logger.exception(f"An unexpected error occurred while processing question '{request.query}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
