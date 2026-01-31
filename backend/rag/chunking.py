### Token chunking with overlap

from typing import List
from transformers import AutoTokenizer

## llama3.1:latest
def chunk_text_LLM(
    text: str,
    chunk_tokens: int = 450,
    overlap_tokens: int = 80,
    model_name: str = "hf-internal-testing/llama-tokenizer"
) -> List[str]:

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        model_max_length=10**9  # 🚀 disable max length warnings
    )

    input_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    chunks = []
    start = 0

    while start < len(input_ids):
        end = start + chunk_tokens
        chunk_ids = input_ids[start:end]

        chunks.append(
            tokenizer.decode(chunk_ids, skip_special_tokens=True)
        )

        start += chunk_tokens - overlap_tokens

    return chunks

import tiktoken
from typing import List

def chunk_text_OpenAI(text: str, chunk_tokens: int = 450, overlap_tokens: int = 80) -> List[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_tokens
        chunk = enc.decode(tokens[start:end])
        chunks.append(chunk)
        start = end - overlap_tokens
        if start < 0:
            start = 0
    return chunks

# Test
"""
text = " ".join(
    ["This is a sentence about LLaMA tokenization."] * 200
)

chunks = chunk_text(text)
print("Chunks:", len(chunks))
"""