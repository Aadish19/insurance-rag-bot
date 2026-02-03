from rag.pdf_to_text  import pdf_to_text
from rag.chunking import chunk_text_LLM
from rag.embed_store import build_and_save_index

## Running Script -> python -m testing.testing_pipeline_with_chunking_and_embed_store

PDF_TXT = pdf_to_text("data/knowledge.pdf")
INDEX_PATH = "vector.index"
META_PATH = "vector_meta.json"


def main():
    # 1. Get chunks from PDF
    chunks = chunk_text_LLM(text = PDF_TXT)
    print(f"Total chunks received: {len(chunks)}")

    # 2. Build FAISS index + store metadata -  this will create files - vector_meta.json and vector.index
    build_and_save_index(
        chunks=chunks,
        index_path=INDEX_PATH,
        meta_path=META_PATH
    )

    print("✅ Embeddings created and index saved")


if __name__ == "__main__":
    main()