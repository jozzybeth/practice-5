import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def build_hybrid_retriever(articles: list[dict], rag_config: dict):
    """
    Qdrant + BM25
    """
    documents = [
        Document(page_content=art["text"], metadata={"id": art["id"], "title": art["title"]})
        for art in articles
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=rag_config["chunk_size"],
        chunk_overlap=rag_config["chunk_overlap"]
    )
    chunks = text_splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(
        model_name=rag_config["embedding_model"],
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )

    db_path = rag_config.get("vector_db_path", "models/qdrant_db")

    if os.path.exists(db_path) and os.listdir(db_path):
        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embedding_model,
            collection_name="corporate_kb",
            path=db_path
        )
    else:
        vector_store = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embedding_model,
            path=db_path,
            collection_name="corporate_kb"
        )

    dense_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": rag_config["top_k_retrieval"]}
    )

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = rag_config["top_k_retrieval"]

    return EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=rag_config["weights"]
    )


def format_docs(docs: list) -> str:
    """Форматирование найденных контекстов в строку."""
    return "\n\n".join([
        f"[Источник {i}] {doc.metadata['title']} (ID статьи: {doc.metadata['id']})\n{doc.page_content}"
        for i, doc in enumerate(docs, 1)
    ])


def get_rag_chain(generator: str, model_config: dict, prompt_template: str):
    """
    Создает RAG-chain
    """
    if generator == "groq":
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
        llm = ChatGroq(
            model=model_config["model_name"],
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"]
        )
    elif generator == "vllm":
        llm = ChatOpenAI(
            model=model_config["model_name"],
            base_url=model_config["base_url"],
            api_key=model_config["api_key"],
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"]
        )
    else:
        raise ValueError(f"Неизвестный генератор {generator}")

    prompt = ChatPromptTemplate.from_template(prompt_template)
    return prompt | llm | StrOutputParser()