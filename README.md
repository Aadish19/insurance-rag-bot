# insurance-rag-bot
Building a RAG agent. A normal chatbot tries to answer from memory or based on Keyword Logic written within your code. But,  A RAG chatbot answers after checking your documents.  Your PDFs are the textbook. RAG is the open-book strategy that makes answers grounded.


## Level 0 - What we’re building!

This product has two parts.

**A backend (Python)** that reads PDFs, breaks them into small pieces, turns them into embeddings, searches them, and generates answers.

**A frontend (React)** that shows a clean chat box on your website like a customer support widget.

When someone asks “How do I file a claim?”, the backend does not guess. It looks inside your PDF knowledge base, grabs the most relevant lines, and answers using that.

## Stack to build the RAG Bot (simple and clean)

**Backend (FastAPI):** This is the brain’s office that listens to questions and sends back answers.

**PDF reading (pypdf):** This is the tool that opens your PDF and pulls the words out of it.

**Chunking (token chunks + overlap):** This cuts the PDF text into small pieces, with a little repeat so nothing important gets cut off.

**Embeddings (text-embedding-3-small):** This turns each text piece into number fingerprints so the computer can find similar meaning fast.

**Vector DB (FAISS):** This is a super-fast storage box that helps you quickly find the best matching text pieces.

**Chat generation (OpenAI Responses API):** This is the writer that reads the best pieces and talks back like a helpful customer support agent.


# Knowledge of some methods used

## 1. Chunking (the most important RAG step)
Chunking means breaking a big document into smaller pieces so search works well.

If chunks are too big, the system grabs long blocks and the answer becomes messy.

If chunks are too small, the system loses meaning and context.

**The most reliable default for real-world PDFs is ###token chunking with overlap.**

Overlap repeats a small part between chunks, so you do not cut an important sentence in half.

------------------------------
**Chunking types you should know (simple one-liners)**
Token chunking: Same size chunks, works for almost every PDF, best default.

Section chunking: Split by headings or blank lines, great for clean docs, breaks on messy docs.

Sentence chunking: Split sentence by sentence, natural flow, but can create too many tiny chunks.

Paragraph chunking: Split by paragraphs, keeps one idea together, and depends on clean formatting.

Fixed character chunking: Cut every X characters, simple and fast, can cut sentences mid-way.

Sliding window chunking: Move a window across text with overlap, great context coverage.

Semantic chunking: Split when the topic changes, understand the context better, and produce the best quality outputs (* this is my favorite type of chunking)

Table-aware chunking: Keeps tables intact so rows do not get scrambled in search.

Hybrid chunking: Mix headings plus token limits so it works even on messy PDFs.

## 2. 
    