import os
import requests
import asyncio
import aiofiles
from pypdf import PdfReader
from typing import List, Dict, Any
from dotenv import load_dotenv
from mistralai.client import MistralClient
import numpy as np

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
            async with requests.get(url, stream=True) as response:
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

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        print(f"Embedding {len(texts)} chunks using Mistral Embed...")
        try:
            embeddings_response = self.mistral_client.embeddings(
                model="mistral-embed",
                input=texts
            )
            embeddings = [data.embedding for data in embeddings_response.data]
            return np.array(embeddings).astype('float32') # FAISS requires float32
        except Exception as e:
            print(f"Error embedding texts with Mistral: {e}")
            raise

    async def process_url_for_embeddings(self, url: str) -> List[Dict[str, Any]]:
        local_file_path = ""
        try:
            local_file_path = await self._download_pdf(url)
            full_text = self._extract_text_from_pdf(local_file_path)
            chunks = self._get_text_chunks(full_text)

            if not chunks:
                print(f"No text chunks extracted from {url}")
                return []

            embeddings = self._embed_texts(chunks)

            indexed_data = []
            for i, chunk in enumerate(chunks):
                indexed_data.append({
                    "embedding": embeddings[i],
                    "text": chunk,
                    "source": f"{url}_chunk_{i}" # Identifier for justification
                })
            return indexed_data
        finally:
            if os.path.exists(local_file_path):
                os.remove(local_file_path)
                print(f"Cleaned up temporary file: {local_file_path}")
