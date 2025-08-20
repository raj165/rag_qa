import uvicorn

if __name__ == "__main__":
    uvicorn.run("rag_qa_system.app:app", host="localhost", port=8000, reload=True)
