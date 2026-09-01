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