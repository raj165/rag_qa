#  RAG QA (PDF/TXT/Website)  

A FastAPI-based **Retrieval-Augmented Generation (RAG)** system that lets you ingest documents (PDF, TXT, Websites) into a **Chroma vector database**, and ask natural language questions answered using **Azure OpenAI GPT models** with strict citation rules.  

---

##  Features
🔹 Ingest **PDF, TXT, MD, LOG files** via REST API  
🔹 Ingest **websites** (URL → clean text)  
🔹 Automatic **chunking** (~1200 tokens, with overlap)  
🔹 Embedding + storage in **Chroma** (local persistent vector DB)  
🔹 Ask questions and get answers with **inline citations**  
🔹 Strict RAG prompt → never hallucinates, returns *“I don’t know”* if not found  
🔹 Sources tracked and returned with every answer  

---