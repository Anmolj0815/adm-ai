import os
import json
import numpy as np
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from ..models.response_models import AdmissionDecisionResponse
from ..services.document_processor import DocumentProcessor
import asyncio # For async operations like document processing

load_dotenv()

class LLMService:
    def __init__(self):
        self.client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
        self.document_processor = DocumentProcessor()

    async def process_admission_query(self, query: str, document_urls: List[str]) -> AdmissionDecisionResponse:
        indexed_documents_data = []
        if document_urls:
            # Dynamically parse, chunk, embed and index documents from provided URLs
            for url in document_urls:
                print(f"Processing document from URL for RAG: {url}")
                try:
                    doc_data = await self.document_processor.process_url_for_embeddings(url)
                    if doc_data:
                        indexed_documents_data.extend(doc_data)
                except Exception as e:
                    print(f"Error processing document from {url}: {e}")

        retrieved_information = self._semantically_retrieve_information(query, indexed_documents_data)

        # Step 1: Parse and structure the query
        parsed_query = self._parse_query_with_llm(query)

        # Step 3: Evaluate and determine decision
        decision, amount, justification, clauses_used = self._evaluate_with_llm(parsed_query, retrieved_information)

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
            model="mistral-embed", # Mistral's embedding model
            input=[query]
        )
        query_embedding = np.array(query_embedding_response.data[0].embedding)

        # Extract embeddings and corresponding texts from indexed_documents_data
        document_embeddings = np.array([d["embedding"] for d in indexed_documents_data])
        document_texts = [d["text"] for d in indexed_documents_data]

        # Ensure embeddings are float32, required by FAISS
        document_embeddings = document_embeddings.astype('float32')
        query_embedding = query_embedding.astype('float32')

        # Create a FAISS index
        dimension = document_embeddings.shape[1]
        import faiss
        index = faiss.IndexFlatL2(dimension)
        index.add(document_embeddings)

        # Perform similarity search
        k = min(5, len(indexed_documents_data)) # Retrieve top K relevant chunks
        distances, indices = index.search(np.expand_dims(query_embedding, axis=0), k)

        relevant_clauses = []
        for i in indices[0]:
            if i != -1: # Ensure index is valid
                relevant_clauses.append(document_texts[i])
        
        if not relevant_clauses:
            return ["No highly relevant information found in the provided documents."]

        print(f"Retrieved {len(relevant_clauses)} relevant clauses using FAISS.")
        return relevant_clauses

    def _evaluate_with_llm(self, parsed_query: Dict[str, Any], relevant_information: List[str]) -> tuple:
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
                model="mistral-large-latest", # Or mistral-small-latest for cost efficiency
                messages=messages,
                response_format={"type": "json_object"}
            )
            llm_output = json.loads(chat_response.choices[0].message.content)
            return (
                llm_output.get("Decision", "Requires Further Review"),
                llm_output.get("Amount"),
                llm_output.get("Justification", "Could not determine a clear justification."),
                llm_output.get("ClausesUsed", [])
            )
        except Exception as e:
            print(f"Error evaluating with Mistral: {e}")
            return "Error", None, "An error occurred during decision evaluation.", []
