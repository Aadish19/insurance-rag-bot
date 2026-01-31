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
