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
