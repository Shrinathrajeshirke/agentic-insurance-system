import json
import uuid
import sys
import os
from src.exception import CustomException
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.config import settings
from src.logger import logger
from src.agents.tools import get_embedding_client, get_qdrant_client, _detect_clause_type


def initialize_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """Creates a Qdrant collection if it does not already exist."""
    collections = client.get_collections().collections
    if not any(col.name == collection_name for col in collections):
        logger.info(f"Creating new Qdrant collection: {collection_name} (dim: {vector_size})")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    else:
        logger.info(f"Collection '{collection_name}' already exists.")


def ingest_policies():
    """Reads sample policies, generates embeddings, and uploads to Qdrant."""
    try:
        data_path = os.path.join(os.getcwd(), "data", "raw_policies", "sample_policies.json")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Policy dataset not found at {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            policies = json.load(f)

        logger.info(f"Loaded {len(policies)} policy clauses from JSON.")

        # Unified embedding & Qdrant client
        embeddings = get_embedding_client()
        client = get_qdrant_client()

        # Dynamic dimension check
        test_vec = embeddings.embed_query("test")
        vector_size = len(test_vec)

        initialize_qdrant_collection(client, settings.QDRANT_COLLECTION_NAME, vector_size)

        points = []
        for policy in policies:
            content = policy.get("content") or policy.get("text", "")
            if not content.strip():
                continue

            vector = embeddings.embed_query(content)

            # Metadata normalization
            metadata = {k: v for k, v in policy.items() if k not in ["content", "text"]}
            metadata["text"] = content
            if "section" not in metadata:
                metadata["section"] = _detect_clause_type(content)
            if "page" not in metadata:
                metadata["page"] = policy.get("page", 1)
            if "insurer" not in metadata:
                metadata["insurer"] = policy.get("insurer", "HDFC Life")

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=metadata
                )
            )

        # Upsert points
        logger.info(f"Uploading {len(points)} vectors to Qdrant collection '{settings.QDRANT_COLLECTION_NAME}'...")
        client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=points
        )
        logger.info("Ingestion completed successfully.")
        print(f"Success: Indexed {len(points)} clauses into '{settings.QDRANT_COLLECTION_NAME}'")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        ingest_policies()
    except Exception as e:
        print(f"Ingestion failed: {e}")