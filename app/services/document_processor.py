import os
from pypdf import PdfReader
from typing import List, Dict, Any
from dotenv import load_dotenv
from mistralai.client import MistralClient
import numpy as np
import faiss
import pickle

load_dotenv()

class DocumentProcessor:
    def __init__(self):
        self.mistral_client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))

    def _extract_text_from_pdf(self, file_path: str) -> str:
        print(f"Extracting text from PDF: {file_path}")
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    def _get_text_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else []
                current_length = sum(len(w) + 1 for w in current_chunk) - 1 if current_chunk else 0

            current_chunk.append(word)
            current_length += len(word) + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def _embed_texts_in_batches(self, texts: List[str], batch_size: int = 250) -> np.ndarray:
        print(f"Embedding {len(texts)} chunks in batches of {batch_size}...")
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                embeddings_response = self.mistral_client.embeddings(
                    model="mistral-embed",
                    input=batch
                )
                embeddings = [data.embedding for data in embeddings_response.data]
                all_embeddings.extend(embeddings)
                print(f"Processed batch {i // batch_size + 1}")
            except Exception as e:
                print(f"Error embedding batch with Mistral: {e}")
                raise
        return np.array(all_embeddings).astype('float32')

    def pre_index_documents(self, data_dir: str = "data/admission_policies", output_path: str = "data/embeddings/admissions_embeddings.pkl"):
        all_chunks = []
        source_map = []
        
        for root, _, files in os.walk(data_dir):
            for file_name in files:
                if file_name.endswith(".pdf"):
                    file_path = os.path.join(root, file_name)
                    full_text = self._extract_text_from_pdf(file_path)
                    chunks = self._get_text_chunks(full_text)
                    all_chunks.extend(chunks)
                    source_map.extend([f"{file_name}_chunk_{i}" for i in range(len(chunks))])
        
        if not all_chunks:
            print("No documents found or no text extracted. Index not created.")
            return

        embeddings_array = self._embed_texts_in_batches(all_chunks)
        
        indexed_data = []
        for i, chunk in enumerate(all_chunks):
            indexed_data.append({
                "embedding": embeddings_array[i],
                "text": chunk,
                "source": source_map[i]
            })

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(indexed_data, f)
        print(f"Successfully created and saved embeddings index to {output_path}")
