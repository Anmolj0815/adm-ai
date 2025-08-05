import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.document_processor import DocumentProcessor

if __name__ == "__main__":
    load_dotenv()
    processor = DocumentProcessor()
    processor.pre_index_documents()
