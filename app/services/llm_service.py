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
        self.indexed_documents_data = [] # Initialize with an empty list

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
        if not indexed_documents_data:
            return ["No documents provided or processed for retrieval."]

        query_embedding_response = self.client.embeddings(
            model="mistral-embed",
            input=[query]
        )
        query_embedding = np.array(query_embedding_response.data[0].embedding)

        document_embeddings = np.array([d["embedding"] for d in indexed_documents_data])
        document_texts = [d["text"] for d in indexed_documents_data]

        document_embeddings = document_embeddings.astype('float32')
        query_embedding = query_embedding.astype('float32')

        dimension = document_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(document_embeddings)

        k = min(5, len(indexed_documents_data))
        distances, indices = index.search(np.expand_dims(query_embedding, axis=0), k)

        relevant_clauses = []
        for i in indices[0]:
            if i != -1:
                relevant_clauses.append(document_texts[i])
        
        if not relevant_clauses:
            return ["No highly relevant information found in the provided documents."]

        print(f"Retrieved {len(relevant_clauses)} relevant clauses using FAISS.")
        return relevant_clauses

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
