import os
import json
import numpy as np
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from ..models.response_models import AdmissionDecisionResponse
from ..services.document_processor import DocumentProcessor
from ..utils.helpers import get_env_variable
import asyncio
import aiohttp

load_dotenv()

class LLMService:
    def __init__(self):
        try:
            self.client = MistralClient(api_key=get_env_variable("MISTRAL_API_KEY"))
            self.n8n_webhook_url = get_env_variable("N8N_WEBHOOK_URL", default_value=None)
        except ValueError as e:
            print(f"Failed to initialize Mistral client: {e}")
            self.client = None

        self.document_processor = DocumentProcessor()

    async def process_admission_query(self, query: str, document_urls: List[str]) -> AdmissionDecisionResponse:
        indexed_documents_data = []
        if document_urls:
            for url in document_urls:
                print(f"Processing document from URL for RAG: {url}")
                try:
                    doc_data = await self.document_processor.process_url_for_embeddings(url)
                    if doc_data:
                        indexed_documents_data.extend(doc_data)
                except Exception as e:
                    print(f"Error processing document from {url}: {e}")

        retrieved_information = self._semantically_retrieve_information(query, indexed_documents_data)

        parsed_query = self._parse_query_with_llm(query)

        decision, amount, justification, clauses_used = await self._evaluate_with_llm(parsed_query, retrieved_information)

        return AdmissionDecisionResponse(
            Decision=decision,
            Amount=amount,
            Justification=justification,
            ClausesUsed=clauses_used
        )
    
    # ... rest of the class methods remain unchanged ...

    async def _trigger_n8n_workflow(self, query: str, justification: str, relevant_docs: list):
        """
        Makes an asynchronous POST request to the n8n webhook.
        """
        webhook_payload = {
            "original_query": query,
            "decision_justification": justification,
            "clauses_used": relevant_docs,
            "action_required": "Follow-up with admissions team"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.n8n_webhook_url, json=webhook_payload, timeout=5) as response:
                    response.raise_for_status()
                    print(f"Successfully triggered n8n workflow. Response status: {response.status}")
        except Exception as e:
            print(f"Error calling n8n webhook: {e}")
