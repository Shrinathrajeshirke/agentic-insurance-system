import sys
import uuid
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.config import settings
from src.logger import logger
from src.exception import CustomException
from src.agents.llm import get_embeddings

@tool
def search_policy_contracts(query: str, clause_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Search indexed term life insurance contracts in the Qdrant vector database.
    Use this tool for exact policy wording, eligibility, wating periods, and exclusions.
    """
    try:
        logger.info(f"Querying Qdrant for: '{query}' (clause_type: {clause_type})")
        client = QdrantClient(host=settings.QDRANT_HOST, 
                              port=settings.QDRANT_PORT,
                              timeout=10, 
                              check_compatibility=False)
        embeddings = get_embeddings()
        query_vector = embeddings.embed_query(query)

        query_filter = None
        if clause_type:
            query_filter = Filter(
                must = [FieldCondition(key="clause_type", match=MatchValue(value=clause_type))]
            )

        results = client.query_points(
            collection_name = settings.QDRANT_COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=3
        )

        formatted_results = []
        for res in results.points:
            formatted_results.append({
                "score": float(res.score),
                "insurer": res.payload.get("insurer"),
                "policy_name": res.payload.get("policy_name"),
                "clause_type": res.payload.get("clause_type"),
                "text": res.payload.get("text")
            })

        logger.info(f"Found {len(formatted_results)} results in Qdrant.")
        return formatted_results

    except Exception as e:
        raise CustomException(e, sys)

@tool 
def web_search(query: str) -> str:
    """
    Search the live web for insurance market trends, missing plans, or claim settlement ratios.
    Use this when internal policy contracts do not cover the requested insurer or topic.
    """
    try:
        logger.info(f"Executing web serach for: '{query}'")
        search_engine = DuckDuckGoSearchResults(num_results=3)
        results = search_engine.run(query)
        return results
    except Exception as e:
        logger.error(f"Web search error: {str(e)}")
        return f"Web search could not retrieve results for '{query}'."

def ingest_policy_document(file_path: str, policy_name: str, insurer: str) -> str:
    """
    Utility to parse a policy PDF/text file, embed chunks, and dynamically add to Qdrant.
    """
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from qdrant_client.models import PointStruct

        logger.info(f"Ingesting uploaded policy document via LangChain loader: {file_path}")

        # Load the PDF as LangChain Document objects
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        if not docs:
            return "Failed to extract readable text from PDF."

        # Split the documents
        splitter = RecursiveCharacterTextSplitter(chunk_size = 700, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        embeddings = get_embeddings()

        points = []
        for doc in chunks:
            # Embed the text content
            vector = embeddings.embed_query(doc.page_content)

            ## Combine custom metadata ewith LangChain's default metadata (source, page)
            payload = {
                "policy_name": policy_name,
                "insurer": insurer,
                "clause_type": "user_uploaded_clause",
                "text": doc.page_content,
                "source": doc.metadata.get("source", file_path),
                "page": doc.metadata.get("page", 0)
            }

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload
                )
            )

        # Batch upsert to Qdrant
        client.upsert(
            collection_name = settings.QDRANT_COLLECTION_NAME,
            points=points
        )

        msg = f"Successfully ingested {len(points)} chunks for {policy_name} ({insurer}) into Qdrant."
        logger.info(msg)
        return msg 

    except Exception as e:
        raise CustomException(e, sys)
    