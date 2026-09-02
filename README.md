# Agentic RAG Insurance Advisory System

An intelligent, stateful term life insurance recommendation engine built with LangGraph, Qdrant, Ollama, and FastAPI.

## Setup & Installation

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate

## Step 1: Base Utilities Setup
To ensure production readiness, custom logging and exception handling modules have been implemented.

* **`src/logger.py`**: Configures dual-logging (console and file-based). Logs are generated dynamically with timestamps in a local `logs/` directory, capturing line numbers, log levels, and execution flow.
* **`src/exception.py`**: Implements a `CustomException` class utilizing Python's `sys` module. It intercepts errors to extract the exact filename, line number, and error description, automatically writing the trace to our logger before halting execution.

## Phase 1: Infrastructure & Environment Setup
* **Vector DB Infrastructure (`docker-compose.yml`)**: Deployed Qdrant with persistent local volume storage (`./data/qdrant_storage`) exposed on ports `6333` (REST) and `6334` (gRPC). Local Ollama service is configured and kept commented out for future offline testing.
* **Environment & Settings Management (`src/config.py` & `.env`)**: Implemented type-safe environment configuration with `pydantic-settings` pointing to Groq Cloud (`llama-3.3-70b-versatile`) and local CPU-efficient embeddings (`sentence-transformers/all-MiniLM-L6-v2`).

## Step 2: Domain Schemas, LLM Factory & Verification
* **Domain Schemas (`src/schema/`)**:
  * `user_profile.py`: Pydantic schema for underwriting parameters (Age, Smoker status, Income, Sum Assured).
  * `policy.py`: Data structure for insurance policy clauses with metadata filtering tags.
  * `state.py`: LangGraph `AgentState` definition tracking messages, profile extraction state, and retrieved context.
* **Model & Embedding Adapters (`src/agents/llm.py`)**: Centralized initialization for Groq LLM (`llama-3.3-70b-versatile`) and CPU-optimized embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
* **Self-Test Script (`test_setup.py`)**: End-to-end verification script testing configurations, schema validation, Qdrant connectivity, embedding vector generation, and Groq API calls.

## Phase 2: Data Ingestion Pipeline
* **Base Dataset (`data/raw_policies/sample_policies.json`)**: Created a structured JSON repository of verified insurance clauses (Eligibility, Exclusions, Riders) for HDFC Life and Max Life.
* **Vector Indexer (`src/ingestion/indexer.py`)**: Implemented an ingestion pipeline using `qdrant_client`. It reads structured policy definitions, generates fixed-length dense vectors via the HuggingFace `all-MiniLM-L6-v2` embedding model, dynamically initializes the Qdrant collection if missing, and performs a batch upsert with extensive metadata payloads (age limits, smoker flags, insurer names) for strict hybrid filtering during retrieval.

## Phase 3: Agentic Tools & Fallback Mechanism
* **Policy Vector Search (`search_policy_contracts`)**: LangChain tool querying the Qdrant vector index with metadata filters (`clause_type`) and cosine score extraction.
* **Fallback Web Retrieval (`web_search`)**: DuckDuckGo integration to handle out-of-distribution queries, market statistics, or unindexed policies.
* **Active Runtime Ingestion (`ingest_policy_document`)**: Dynamic parser that splits uploaded PDF contracts using `RecursiveCharacterTextSplitter` and embeds them into Qdrant on the fly.