from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from uuid import uuid4
import shutil
import pathlib
from rag_qa_system.loaders import load_pdf, load_txt, load_site
from rag_qa_system.rag_core import add_documents, answer
load_dotenv()


app = FastAPI(title="RAG QA (PDF/TXT/Website)")

DATA_DIR = pathlib.Path("../../data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

class AskRequest(BaseModel):
    question: str

@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    ext = pathlib.Path(file.filename).suffix.lower()
    tmp_path = DATA_DIR / f"upload_{uuid4().hex}{ext}"
    with tmp_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    if ext in [".pdf"]:
        docs = load_pdf(str(tmp_path))
        source = f"file:{file.filename}"
    elif ext in [".txt", ".md", ".log"]:
        docs = load_txt(str(tmp_path))
        source = f"file:{file.filename}"
    else:
        return JSONResponse(status_code=400, content={"error": f"Unsupported file type: {ext}"})

    n = add_documents(docs, source)
    return {"ingested_chunks": n, "source": source}

@app.post("/ingest/url")
async def ingest_url(url: str = Form(...)):
    docs = load_site(url)
    n = add_documents(docs, source_label=url)
    return {"ingested_chunks": n, "source": url}

@app.post("/ask")
async def ask(req: AskRequest):
    content, docs = answer(req.question)
    citations: List[str] = []
    for d in docs:
        src = d.metadata.get("source")
        if src and src not in citations:
            citations.append(src)
    return {"answer": content, "sources": citations}

@app.get("/")
async def root():
    return {"status": "ok"}