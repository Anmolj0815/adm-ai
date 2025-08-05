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
import faiss
import pickle

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
        self.indexed_documents_data = []

    async def process_admission_query(self, query: str) -> AdmissionDecisionResponse:
        """
        TEMPORARY DIAGNOSTIC METHOD:
        This method skips the FAISS retrieval step and sends the entire document text
        as context to the LLM to check if the PDF text extraction is working correctly.
        """
        full_document_text = ""
        data_dir = "data/admission_policies"
        if os.path.exists(data_dir):
            for file_name in os.listdir(data_dir):
                if file_name.endswith(".pdf"):
                    file_path = os.path.join(data_dir, file_name)
                    full_document_text += self.document_processor._extract_text_from_pdf(file_path)
        
        # We put the entire document text into a list to match the expected input for _evaluate_with_llm
        retrieved_information = [full_document_text]
        parsed_query = self._parse_query_with_llm(query)
        decision, amount, justification, clauses_used = await self._evaluate_with_llm(parsed_query, retrieved_information)
        return AdmissionDecisionResponse(
            Decision=decision,
            Amount=amount,
            Justification=justification,
            ClausesUsed=clauses_used
        )

    def _parse_query_with_llm(self, query: str) -> Dict[str, Any]:
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant that extracts key details from admission queries."),
            ChatMessage(role="user", content=f"Extract the following details from this admission query: 'program_name', 'eligibility_criteria', 'location', 'application_deadlines', 'fees_scholarships'. If a detail is not present, mark it as 'N/A'. Query: {query}\n\nReturn in JSON format."),
        ]
        try:
            chat_response = self.client.chat(
                model="mistral-large-latest",
                messages=messages,
                response_format={"type": "json_object"}
            )
            parsed_details = json.loads(chat_response.choices[0].message.content)
            return parsed_details
        except Exception as e:
            print(f"Error parsing query with Mistral: {e}")
            return {"query_raw": query, "program_name": "N/A", "eligibility_criteria": "N/A", "location": "N/A", "application_deadlines": "N/A", "fees_scholarships": "N/A"}

    def _semantically_retrieve_information(self, query: str, indexed_documents_data: List[Dict[str, Any]]) -> List[str]:
        # This method is not used in the temporary fix above.
        return ["RAG retrieval is temporarily disabled for diagnostic purposes."]

    async def _evaluate_with_llm(self, parsed_query: Dict[str, Any], relevant_information: List[str]) -> tuple:
        combined_context = "\n".join(relevant_information)

        prompt = f"""
        You are an AI assistant designed to act as an expert admissions officer for Indian Institute of Management Mumbai. Your task is to accurately answer a candidate's query and make a decision based **only** on the provided 'Relevant Information' from the official admission policy.

        CRITICAL INSTRUCTIONS:
        1.  **Strictly use the provided context.** Do not use any external knowledge.
        2.  **Be specific and factual.** Do not make assumptions or generalize.
        3.  **For Eligibility:** Provide specific percentages, degree requirements, and deadlines.
        4.  **For Final Selection:** Provide the parameters and their exact percentage weights.
        5.  **If the answer is not in the context,** state: "I cannot answer this question based on the provided documents."

        The query details have been pre-parsed for you, and relevant document clauses have been retrieved.

        Query Details:
        {json.dumps(parsed_query, indent=2)}

        Relevant Information from Policy Documents:
        {combined_context}

        Now, based on the above, provide a decision and justification in the following JSON format:
        {{
            "Decision": "string",
            "Amount": "number or null",
            "Justification": "string explaining the decision based on the relevant information. Mention specific percentages, dates, or criteria.",
            "ClausesUsed": ["list of key phrases, sentences, or paragraphs from the 'Relevant Information' that support the justification."]
        }}
        """
        messages = [
            ChatMessage(role="system", content="You are a helpful and precise admissions officer for IIM Mumbai."),
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
