import os
import requests
import asyncio
import aiofiles
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
        self.temp_dir = "data/temp_docs"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.mistral_client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))

    async def _download_pdf(self, url: str) -> str:
        local_filename = os.path.join(self.temp_dir, url.split('/')[-1].split('?')[0])
        if not local_filename.endswith(".pdf"):
            local_filename += ".pdf"
        print(f"Downloading PDF from {url} to {local_filename}")
        
        async with aiofiles.open(local_filename, 'wb') as f:
            response = await asyncio.to_thread(requests.get, url, stream=True)
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192):
                await f.write(chunk)
        print(f"Downloaded: {local_filename}")
        return local_filename

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

    async def process_url_for_embeddings(self, url: str) -> List[Dict[str, Any]]:
        local_file_path = ""
        try:
            local_file_path = await self._download_pdf(url)
            full_text = self._extract_text_from_pdf(local_file_path)
            chunks = self._get_text_chunks(full_text)

            if not chunks:
                print(f"No text chunks extracted from {url}")
                return []

            embeddings = self._embed_texts_in_batches(chunks)

            indexed_data = []
            for i, chunk in enumerate(chunks):
                indexed_data.append({
                    "embedding": embeddings[i],
                    "text": chunk,
                    "source": f"{url}_chunk_{i}"
                })
            return indexed_data
        finally:
            if os.path.exists(local_file_path):
                os.remove(local_file_path)
                print(f"Cleaned up temporary file: {local_file_path}")
