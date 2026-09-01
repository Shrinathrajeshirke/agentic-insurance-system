import sys
from qdrant_client import QdrantClient
from src.config import settings
from src.logger import logger
from src.exception import CustomException
from src.schema.user_profile import UserProfile
from src.schema.policy import PolicyMetadata, PolicyChunk
from src.agents.llm import get_llm, get_embeddings

def test_pipeline():
    print("\n--- 1. Testing Configuration & Logging ---")
    logger.info("Initializing system self-test...")
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Model Provider: {settings.MODEL_PROVIDER}")
    print("Configuration loaded successfully.")

    print("\n--- 2. Testing Pydantic Schemas ---")
    profile = UserProfile(
        age=30, 
        gender="Male",
        annual_income=1200000.0,
        is_smoker=False,
        medical_conditions=["None"],
        dependents_count=2,
        desired_sum_assured=15000000.0,
        policy_term_years=35
    )
    metadata = PolicyMetadata(
        policy_name="Shield Term Plus",
        insurer="SafeLife",
        clause_type="eligibility",
        entry_age_min=18,
        entry_age_max=60,
        smoker_allowed=True,
        riders_available=["Critical Illness", "Accidental Death"]
    )
    chunk = PolicyChunk(chunk_id="chunk_001", content="Sample clause text", metadata=metadata)
    print(f"UserProfile instantiated: Age {profile.age}, Income: {profile.annual_income}")
    print(f"PolicyChunk instantiated: {chunk.metadata.policy_name} ({chunk.metadata.clause_type})")

    print("\n--- 3. Testing Qdrant DB Connection ---")
    try:
        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        collections = client.get_collections()
        print(f"Qdrant reachable! Collections count: {len(collections.collections)}")
    except Exception as e:
        print("Warning: Qdrant is unreachable. Ensure docker container is running (docker-compose up -d)")
        raise CustomException(e, sys)

    print("\n--- 4. Testing Local Embeddings ---")
    embeddings = get_embeddings()
    test_vec = embeddings.embed_query("Term life insurance policy eligibility")
    print(f"Embeddings generated successfully. Vector dimension: {len(test_vec)}")

    print("\n--- 5. Testing Groq Cloud LLM ---")
    llm = get_llm()
    response = llm.invoke("Respond with the single word 'CONNECTED' if you receive this message.")
    print(f"Groq LLM Response: {response.content.strip()}")

if __name__ == "__main__":
    try:
        test_pipeline()
    except CustomException as ce:
        print(f"\nExecution Failed: {ce}")
    except Exception as e:
        print(f"\nUnexpected Failure: {e}")