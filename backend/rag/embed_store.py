'''Embeddings are what make “meaning search” possible. Instead of searching for exact keywords, embeddings let you search by similarity.

Here is the flow:

Convert each chunk into an embedding vector (a list of numbers).

Store those vectors in FAISS (vector DB).

Later, convert the user question into a vector too.

Ask FAISS: which chunk vectors are closest to this question vector?'''

import json
import os
import numpy as np
import faiss
import ollama


## For open AI 
# from openai import OpenAI
# EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
# 
'''
Flow for OpenAI
Text chunks
   ↓
OpenAI embedding model (cloud)
   ↓
Vectors (1536-D)
   ↓
L2 normalize
   ↓
FAISS IndexFlatIP
   ↓
Save index + metadata
'''

'''
client = OpenAI()
def embed_texts_OpenAI(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vectors = [d.embedding for d in resp.data]
    arr = np.array(vectors, dtype="float32")
    faiss.normalize_L2(arr)
    return arr
'''
## Using Ollama callable embedding model : nomic-embed-text 
''' It comes with 
Model	             Quality	 Speed	  Dim
nomic-embed-text	⭐⭐⭐⭐	Fast	768
'''
EMBED_MODEL = "nomic-embed-text"  # local Ollama model

'''
embed_texts_LLM working
Chunks
  ↓
nomic-embed-text (local)
  ↓
768-dim vectors
  ↓
FAISS IndexFlatIP
  ↓
Fast cosine similarity
'''

'''
Full code working with LLM

User query
   ↓
Embed query (nomic-embed-text)
   ↓
FAISS similarity search
   ↓
Top-k chunks
   ↓
Prompt llama3.1 with context
   ↓
Answer
'''

def embed_texts_LLM(texts: list[str]) -> np.ndarray:
    """
    Generate embeddings locally using Ollama
    """
    vectors = []

    for text in texts:
        resp = ollama.embeddings(
            model=EMBED_MODEL,
            prompt=text
        )
        vectors.append(resp["embedding"])

    arr = np.array(vectors, dtype="float32")

    # Normalize for cosine similarity via Inner Product
    faiss.normalize_L2(arr)
    return arr

def build_and_save_index(chunks: list[str], index_path: str, meta_path: str):
    vectors = embed_texts_LLM(chunks)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks}, f, ensure_ascii=False, indent=2)

def load_index(index_path: str, meta_path: str):
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta["chunks"]    