from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_loaders.async_html import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.documents import Document

def load_pdf(path: str) -> List[Document]:
    loader = PyPDFLoader(path)
    return loader.load()


def load_txt(path: str) -> List[Document]:
    # utf-8 with fallback
    try:
        loader = TextLoader(path, encoding="utf-8")
        return loader.load()
    except Exception:
        loader = TextLoader(path, encoding="latin-1")
        return loader.load()

def load_site(url: str) -> List[Document]:
    # Use Async loader (fetches raw HTML)
    loader = AsyncHtmlLoader(url)
    docs = loader.load()

    # Convert HTML → plain text
    transformer = Html2TextTransformer()
    docs_transformed = transformer.transform_documents(docs)

    return docs_transformed