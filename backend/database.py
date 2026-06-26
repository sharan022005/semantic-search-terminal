import motor.motor_asyncio
import chromadb
from backend import config

# MongoDB client and collections
mongo_client = motor.motor_asyncio.AsyncIOMotorClient(config.MONGODB_URL)
db = mongo_client[config.MONGODB_DB_NAME]
documents_metadata = db["documents_metadata"]

# ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIRECTORY)
chroma_collection = chroma_client.get_or_create_collection(
    name=config.CHROMA_COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"} # Use cosine similarity space to match normal SentenceTransformers behavior
)
