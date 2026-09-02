import json
import uuid
import sys
import os
from src.exception import CustomException
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.config import settings
from src.logger import logger
from src.agents.llm import get_embeddings

def initialize_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """Creates a Qdrant collection if it does not already exist."""
    collections = client.get_collections().collections
    if not any(col.name == collection_name for col in collections):
        logger.info(f"Creating new Qdrant collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config = VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    else:
        logger.info(f"Collection '{collection_name}' already exists. Appending data.")

def ingest_policies():
    """Reads sample policies, generate embeddings, and uploads to Qdrant."""
    try:
        data_path = os.path.join(os.getcwd(), "data", "raw_policies", "sample_policies.json")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Database not found at {data_path}")
        with open(data_path, "r") as f:
            policies = json.load(f)

        logger.info(f"Loaded {len(policies)} policy clauses from JSON.")

        ## Initialize local embeddings
        embeddings = get_embeddings()

        ## Determine vector size dynamically based on the embedding model
        test_vec = embeddings.embed_query("test")
        vector_size = len(test_vec)

        ## Connect to Qdrant
        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

        initialize_qdrant_collection(client, settings.QDRANT_COLLECTION_NAME, vector_size)

        points = []

        for policy in policies:
            content = policy["content"]
            vector = embeddings.embed_query(content)

            ## Remove content from metadata to keep payload clean, we store it in a dedicated field.
            metadata = {k: v for k,v in policy.items() if k != "content"}
            metadata["text"] = content

            point = PointStruct(
                id = str(uuid.uuid4()),
                vector = vector,
                payload = metadata
            )
            points.append(point)

        # Upload to Qdrant in batches
        logger.info("Uploading vectors to Qdrant...")
        client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=points
        )
        logger.info(f"Successfully indexed {len(points)} policy chunks into Qdrant.")
        print(f"Success: Indexed {len(points)} clauses into collection '{settings.QDRANT_COLLECTION_NAME}'")

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        ingest_policies()
    except Exception as e:
        print(f"Ingestion failed: {e}")