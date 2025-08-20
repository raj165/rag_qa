import os
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", "../data/chroma")
TOP_K = int(os.getenv("TOP_K", 5))

def get_embeddings():
    return AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
    model=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
    )


def get_llm():
    return AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
    model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    temperature=0.0,
    )

# Shared splitter
SPLITTER = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
            )


def chunks_from_documents(docs: List[Document]) -> List[Document]:
    return SPLITTER.split_documents(docs)


def get_or_create_vectorstore():
    embeddings = get_embeddings()
    vs = Chroma(collection_name="rag_collection", embedding_function=embeddings, persist_directory=CHROMA_DIR)
    return vs

def add_documents(docs: List[Document], source_label: str) -> int:
    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata.setdefault("source", source_label)
        chunks = chunks_from_documents(docs)
        vs = get_or_create_vectorstore()
        vs.add_documents(chunks)
        return len(chunks)

# Prompt with citation markers
PROMPT = ChatPromptTemplate.from_messages([
        ("system", (
        "You are a strict RAG assistant. Answer ONLY from the provided context. "
        "If the answer is not in the context, say 'I don't know based on the provided sources.'\n"
        "Cite sources as [source] after the sentence they support."
        )),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer in a few sentences with inline citations.")
        ])

def _format_docs(docs: List[Document]) -> str:
    lines = []
    for d in docs:
        src = d.metadata.get("source")
        # include short preview
        text = (d.page_content or "").strip().replace("\n", " ")
        preview = text[:500]
        lines.append(f"[source: {src}] {preview}")
        return "\n\n".join(lines)


def retrieve(question: str) -> List[Document]:
    vs = get_or_create_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
    return retriever.invoke(question)


def answer(question: str) -> Tuple[str, List[Document]]:
    docs = retrieve(question)
    context = _format_docs(docs)
    llm = get_llm()
    chain = PROMPT | llm
    resp = chain.invoke({"question": question, "context": context})
    return resp.content, docs