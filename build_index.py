"""
build_index.py – one-shot script to build the FAISS index.

Run this locally before deploying to Render so the index is
already on disk and no embedding calls are made at runtime.

Usage:
    python build_index.py
"""

import sys
import os

# Make sure we run from project root
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from backend.services.vector_service import VectorService

if __name__ == "__main__":
    print("Building FAISS index from knowledge base…")
    vs = VectorService()
    vs.load_or_build_index()
    print(f"✅ Done. Index saved to vector_store/")
    print(f"   Vectors: {vs._index.ntotal}")
