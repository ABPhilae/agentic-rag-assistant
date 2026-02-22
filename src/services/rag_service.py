from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.config import get_settings
import uuid


async def index_document(file_path: str, filename: str) -> int:
    settings = get_settings()
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata['source'] = filename
    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=settings.openai_api_key
    )
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    # Ensure collection exists
    try:
        client.get_collection(settings.qdrant_collection)
    except Exception:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
    points = []
    for chunk in chunks:
        vector = embeddings.embed_query(chunk.page_content)
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={'page_content': chunk.page_content, **chunk.metadata}
        ))
    if points:
        client.upsert(collection_name=settings.qdrant_collection, points=points)
    return len(points)
