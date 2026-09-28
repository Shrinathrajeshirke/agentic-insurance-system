import sys
import os
import json
import uuid
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from duckduckgo_search import DDGS
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest

from src.config import settings
from src.logger import logger
from src.exception import CustomException

# Managed remote embeddings via HF Serverless API (Zero local RAM footprint)
_embeddings = None
_ranker = None
_bm25_index = None
_bm25_corpus = []


def get_embedding_client():
    global _embeddings
    if _embeddings is None:
        logger.info("Initializing HuggingFace Inference API embeddings client...")
        _embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction",
            huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN,
        )
    return _embeddings


def get_qdrant_client() -> QdrantClient:
    if settings.QDRANT_URL and settings.QDRANT_API_KEY:
        return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=10.0)
    return QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=5.0)


def get_flashranker() -> Ranker:
    global _ranker
    if _ranker is None:
        logger.info("Initializing lightweight FlashRank cross-encoder reranker...")
        _ranker = Ranker()
    return _ranker


def _init_bm25():
    """Initializes in-memory BM25 index over policy corpus."""
    global _bm25_index, _bm25_corpus
    if _bm25_index is not None:
        return

    data_path = os.path.join(os.getcwd(), "data", "raw_policies", "sample_policies.json")
    if not os.path.exists(data_path):
        return

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            _bm25_corpus = json.load(f)

        tokenized_corpus = [
            f"{doc.get('section', '')} {doc.get('insurer', '')} {doc.get('content', '')}".lower().split()
            for doc in _bm25_corpus
        ]
        _bm25_index = BM25Okapi(tokenized_corpus)
        logger.info(f"Initialized BM25 index with {len(_bm25_corpus)} policy clauses.")
    except Exception as e:
        logger.warning(f"Failed to initialize BM25 index: {e}")


def _detect_clause_type(text: str) -> str:
    lower = text.lower()
    if any(k in lower for k in ["suicide", "exclusion", "not covered", "exclusions"]):
        return "Exclusions & Restrictions"
    if any(k in lower for k in ["grace period", "revival", "lapse"]):
        return "Grace Period & Revival"
    if any(k in lower for k in ["free look", "cancellation"]):
        return "Free-Look Period"
    if any(k in lower for k in ["rider", "critical illness", "accidental death"]):
        return "Riders & Add-on Benefits"
    if any(k in lower for k in ["surrender", "special exit", "paid up"]):
        return "Surrender & Exit Value"
    if any(k in lower for k in ["hlv", "human life value", "multiplier", "income limit"]):
        return "Underwriting & HLV"
    if any(k in lower for k in ["gst", "tax", "80c"]):
        return "Taxation & GST"
    if any(k in lower for k in ["claim", "settlement", "nominee"]):
        return "Claims & Payout Provisions"
    return "Terms & Conditions"


@tool
def search_policy_contracts(query: str, clause_type: str = None) -> list:
    """Hybrid Search (Dense Vector + BM25 Lexical + Cross-Encoder Rerank) over policy contracts."""
    try:
        logger.info(f"Querying Hybrid Search for: '{query}'")
        _init_bm25()

        client = get_qdrant_client()
        embeddings = get_embedding_client()

        # 1. Dense Retrieval via Qdrant
        query_vector = embeddings.embed_query(query)
        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query=query_vector,
                limit=8
            )
            dense_hits = response.points
        else:
            dense_hits = client.search(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query_vector=query_vector,
                limit=8
            )

        candidate_map: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = {}

        # Accumulate Dense RRF scores (k=60)
        for rank, hit in enumerate(dense_hits):
            payload = hit.payload or {}
            text = payload.get("text", "")
            if not text:
                continue
            doc_id = str(hit.id)
            candidate_map[doc_id] = {
                "insurer": payload.get("insurer", "Unknown Insurer"),
                "policy_name": payload.get("policy_name", payload.get("insurer", "Policy Contract")),
                "page": payload.get("page", 1),
                "section": payload.get("section", "General"),
                "clause_type": payload.get("clause_type", "clause"),
                "text": text,
            }
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (60 + rank + 1))

        # 2. Sparse Lexical Retrieval via BM25
        if _bm25_index and _bm25_corpus:
            tokenized_query = query.lower().split()
            bm25_scores = _bm25_index.get_scores(tokenized_query)
            top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:8]

            for rank, idx in enumerate(top_bm25_indices):
                if bm25_scores[idx] <= 0:
                    continue
                doc = _bm25_corpus[idx]
                doc_id = f"bm25_{idx}"
                text = doc.get("content") or doc.get("text", "")
                if doc_id not in candidate_map:
                    candidate_map[doc_id] = {
                        "insurer": doc.get("insurer", "Policy Insurer"),
                        "policy_name": doc.get("insurer", "Policy Contract"),
                        "page": doc.get("page", 1),
                        "section": doc.get("section", _detect_clause_type(text)),
                        "clause_type": doc.get("section", _detect_clause_type(text)),
                        "text": text,
                    }
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (60 + rank + 1))

        if not candidate_map:
            return []

        # Sort candidates by combined RRF score
        sorted_candidate_ids = sorted(candidate_map.keys(), key=lambda did: rrf_scores.get(did, 0.0), reverse=True)[:10]
        passages = [
            {"id": cid, "text": candidate_map[cid]["text"]}
            for cid in sorted_candidate_ids
        ]

        # 3. Cross-Encoder Re-ranking via FlashRank
        try:
            ranker = get_flashranker()
            rerank_req = RerankRequest(query=query, passages=passages)
            reranked_results = ranker.rerank(rerank_req)
            top_reranked_ids = [r["id"] for r in reranked_results[:4]]
        except Exception as r_err:
            logger.warning(f"Reranker failed, falling back to RRF ordering: {r_err}")
            top_reranked_ids = sorted_candidate_ids[:4]

        final_results = []
        for cid in top_reranked_ids:
            item = candidate_map[cid]
            final_results.append({
                "insurer": item["insurer"],
                "policy_name": item["policy_name"],
                "page": item["page"],
                "section": item["section"],
                "clause_type": item["clause_type"],
                "text": item["text"],
                "score": round(rrf_scores.get(cid, 1.0), 3)
            })

        return final_results
    except Exception as e:
        logger.error(f"CRITICAL: Hybrid search error: {e}", exc_info=True)
        return []


@tool
def web_search(query: str) -> str:
    """Fallback web search using DuckDuckGo with regional market pricing benchmarks."""
    search_query = f"{query} term life insurance premium India IRDAI"
    try:
        logger.info(f"DuckDuckGo search for: '{search_query}'")
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, region="in-en", max_results=3))
            if results:
                return "\n---\n".join([f"Title: {r.get('title')}\nSnippet: {r.get('body')}" for r in results])
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")

    return (
        "Indian Market Term Insurance Benchmark Rates (IRDAI Insurers, 28-30M, Non-Smoker, 1 Cr Cover, 30-35 Yr Term):\n"
        "1. HDFC Life Click 2 Protect Super: Base premium ~₹950 - ₹1,250/month (~₹11,000 - ₹14,500/year). Claim Settlement Ratio: 99.3%.\n"
        "2. Max Life Smart Secure Plus: Base premium ~₹850 - ₹1,150/month (~₹10,000 - ₹13,500/year). Features Special Exit Value (zero-cost exit). Claim Settlement Ratio: 99.65%.\n"
        "3. Tata AIA Sampoorna Raksha Supreme: Base premium ~₹900 - ₹1,200/month (~₹10,500 - ₹14,000/year). Up to 40 critical illness coverage options. Claim Settlement Ratio: 99.1%.\n"
        "4. ICICI Prudential iProtect Smart: Base premium ~₹980 - ₹1,300/month (~₹11,500 - ₹15,000/year). Automatic waiver of premium on permanent disability. Claim Settlement Ratio: 98.9%.\n"
        "Return of Premium (TROP) Option: Typically increases base premium by ~2.2x to 2.3x."
    )


def ingest_policy_document(file_path: str, policy_name: str, insurer: str) -> str:
    """Parses an uploaded PDF policy brochure, captures page numbers, and indexes into Qdrant."""
    try:
        logger.info(f"Ingesting PDF: {file_path} for policy {policy_name} ({insurer})")
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        if not docs:
            return "Failed to extract text from the provided document."

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=120
        )
        split_docs = text_splitter.split_documents(docs)
        client = get_qdrant_client()
        embeddings = get_embedding_client()

        texts = [doc.page_content.strip() for doc in split_docs if doc.page_content.strip()]
        if not texts:
            return "No valid text chunks were found in the uploaded file."

        vectors = embeddings.embed_documents(texts)

        points = []
        valid_idx = 0
        for doc in split_docs:
            chunk_text = doc.page_content.strip()
            if not chunk_text:
                continue

            page_num = doc.metadata.get("page", 0) + 1
            detected_sec = _detect_clause_type(chunk_text)

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vectors[valid_idx],
                    payload={
                        "policy_name": policy_name,
                        "insurer": insurer,
                        "page": page_num,
                        "section": detected_sec,
                        "clause_type": detected_sec,
                        "text": chunk_text
                    }
                )
            )
            valid_idx += 1

        if points:
            client.upsert(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                points=points
            )
            return f"Successfully indexed {len(points)} chunks with page tracking into Qdrant."

        return "No points generated."
    except Exception as e:
        raise CustomException(e, sys)