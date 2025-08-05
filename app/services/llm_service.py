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
import aiohttp # For making async HTTP requests

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

        # The rest of this function remains the same
        retrieved_information = self._semantically_retrieve_information(query, indexed_documents_data)
        parsed_query = self._parse_query_with_llm(query)
        decision, amount, justification, clauses_used = await self._evaluate_with_llm(parsed_query, retrieved_information)

        return AdmissionDecisionResponse(
            Decision=decision,
            Amount=amount,
            Justification=justification,
            ClausesUsed=clauses_used
        )

    # ... _parse_query_with_llm and _semantically_retrieve_information remain unchanged ...

    async def _evaluate_with_llm(self, parsed_query: Dict[str, Any], relevant_information: List[str]) -> tuple:
        combined_context = "\n".join(relevant_information)

        prompt = f"""
        Given the following admission query details and relevant information extracted from admission policies, determine the admission decision (Approved, Rejected, Requires Further Review), the applicable amount (e.g., application fee, scholarship), and a justification, referencing the specific points/clauses from the 'Relevant Information'.

        Query Details: {json.dumps(parsed_query, indent=2)}
        Relevant Information:
        {combined_context}

        If an applicant meets all criteria based on the relevant information, the decision is 'Approved'. If they clearly do not meet criteria, it's 'Rejected'. If more information is needed or the information is insufficient to make a definite decision, it's 'Requires Further Review'.

        Provide the response in the following JSON format:
        {{
            "Decision": "string",
            "Amount": "number or null",
            "Justification": "string explaining the decision based on relevant information, citing specific details/clauses.",
            "ClausesUsed": ["list of key phrases or summarized clause identifiers used"]
        }}
        """
        messages = [
            ChatMessage(role="system", content="You are an AI assistant for admission inquiries. Provide clear decisions and justifications based on provided information. Be concise and precise."),
            ChatMessage(role="user", content=prompt)
        ]
        try:
            chat_response = self.client.chat(
                model="mistral-large-latest",
                messages=messages,
                response_format={"type": "json_object"}
            )
            llm_output = json.loads(chat_response.choices[0].message.content)
            
            decision = llm_output.get("Decision", "Requires Further Review")
            
            # Integrate n8n webhook call here based on the decision
            if decision == "Requires Further Review" and self.n8n_webhook_url:
                await self._trigger_n8n_workflow(
                    query=parsed_query.get("query_raw", ""),
                    justification=llm_output.get("Justification", ""),
                    relevant_docs=llm_output.get("ClausesUsed", [])
                )

            return (
                decision,
                llm_output.get("Amount"),
                llm_output.get("Justification", "Could not determine a clear justification."),
                llm_output.get("ClausesUsed", [])
            )
        except Exception as e:
            print(f"Error evaluating with Mistral: {e}")
            return "Error", None, "An error occurred during decision evaluation.", []

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
