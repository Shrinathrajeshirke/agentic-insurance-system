# Agentic RAG Insurance Advisory System

An intelligent, stateful term life insurance recommendation engine built with LangGraph, Qdrant, Ollama, and FastAPI[cite: 2].

## Setup & Installation

1. **Create and activate a virtual environment[cite: 2]:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```[cite: 2]

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Step 1: Base Utilities Setup
To ensure production readiness, custom logging and exception handling modules have been implemented[cite: 2].

* **`src/logger.py`**: Configures dual-logging (console and file-based)[cite: 2]. Logs are generated dynamically with timestamps in a local `logs/` directory, capturing line numbers, log levels, and execution flow[cite: 2].
* **`src/exception.py`**: Implements a `CustomException` class utilizing Python's `sys` module[cite: 2]. It intercepts errors to extract the exact filename, line number, and error description, automatically writing the trace to our logger before halting execution[cite: 2].

---

## Phase 1: Infrastructure & Environment Setup
* **Vector DB Infrastructure (`docker-compose.yml`)**: Deployed Qdrant with persistent local volume storage (`./data/qdrant_storage`) exposed on ports `6333` (REST) and `6334` (gRPC)[cite: 2]. Local Ollama service is configured and kept commented out for future offline testing[cite: 2].
* **Environment & Settings Management (`src/config.py` & `.env`)**: Implemented type-safe environment configuration with `pydantic-settings` pointing to Groq Cloud (`llama-3.3-70b-versatile`) and local CPU-efficient embeddings (`sentence-transformers/all-MiniLM-L6-v2`)[cite: 2].

---

## Step 2: Domain Schemas, LLM Factory & Verification
* **Domain Schemas (`src/schema/`)[cite: 2]**:
  * `user_profile.py`: Pydantic schema for underwriting parameters (Age, Smoker status, Income, Sum Assured)[cite: 2].
  * `policy.py`: Data structure for insurance policy clauses with metadata filtering tags[cite: 2].
  * `state.py`: LangGraph `AgentState` definition tracking messages, profile extraction state, and retrieved context[cite: 2].
* **Model & Embedding Adapters (`src/agents/llm.py`)**: Centralized initialization for Groq LLM (`llama-3.3-70b-versatile`) and CPU-optimized embeddings (`sentence-transformers/all-MiniLM-L6-v2`)[cite: 2].
* **Self-Test Script (`test_setup.py`)**: End-to-end verification script testing configurations, schema validation, Qdrant connectivity, embedding vector generation, and Groq API calls[cite: 2].

---

## Phase 2: Data Ingestion Pipeline
* **Base Dataset (`data/raw_policies/sample_policies.json`)**: Created a structured JSON repository of verified insurance clauses (Eligibility, Exclusions, Riders) for HDFC Life and Max Life[cite: 2].
* **Vector Indexer (`src/ingestion/indexer.py`)**: Implemented an ingestion pipeline using `qdrant_client`[cite: 2]. It reads structured policy definitions, generates fixed-length dense vectors via the HuggingFace `all-MiniLM-L6-v2` embedding model, dynamically initializes the Qdrant collection if missing, and performs a batch upsert with extensive metadata payloads (age limits, smoker flags, insurer names) for strict hybrid filtering during retrieval[cite: 2].

---

## Phase 3: Agentic Tools & Fallback Mechanism
* **Policy Vector Search (`search_policy_contracts`)**: LangChain tool querying the Qdrant vector index with metadata filters (`clause_type`) and cosine score extraction[cite: 2].
* **Fallback Web Retrieval (`web_search`)**: DuckDuckGo integration to handle out-of-distribution queries, market statistics, or unindexed policies[cite: 2].
* **Active Runtime Ingestion (`ingest_policy_document`)**: Dynamic parser that splits uploaded PDF contracts using `RecursiveCharacterTextSplitter` and embeds them into Qdrant on the fly[cite: 2].

---

## Phase 4: Stateful Agent Orchestration (LangGraph)
* **`src/agents/graph.py`**: Built an end-to-end `StateGraph` agent workflow[cite: 2]:
  * `profiler_node`: Extracts structured applicant metadata (`UserProfile`) across conversational turns[cite: 2].
  * `retriever_node`: Performs similarity evaluation against Qdrant, falling back to DuckDuckGo search when relevance falls below threshold[cite: 2].
  * `advisor_node`: Synthesizes the extracted profile with retrieved contractual clauses to provide grounded recommendations with citations[cite: 2].

---

## Phase 5: Privacy, Persistence, UI & Evaluation

* **Data Privacy & Sanitization (`src/security/sanitizer.py`)**:
  * Implemented pure-Python regex pattern matching to mask structured identifiers (Credit Cards, SSNs, Emails, Phone numbers, Dates of Birth, ZIP codes) in incoming queries.
  * Direct dictionary key scrubbing to wipe abstract, sensitive traits (Medical records, Race, Religion, Full Name, Gender) before writing state to logs or internal storage.

* **FastAPI Service Layer (`src/api/server.py`)**:
  * REST API exposing `/chat` for stateful multi-turn interactions and `/upload_brochure` for multipart PDF ingestion.
  * Passes incoming requests to LangGraph checkpoints using `thread_id` to maintain conversation memory across disconnections.

* **Streamlit Web Application (`app.py`)**:
  * **Underwriting Onboarding Gate**: Collects baseline eligibility inputs—including salary tiers in Lakhs (`"3 - 6 Lakhs"`, `"6 - 10 Lakhs"`) and desired life cover in Crores (`"1 Crore"`, `"2 Crore"`)—locking the chat interface until the profile assessment is completed.
  * **Top 3 Indian Plan Starter**: Automatically generates the top 3 best-fitting Indian market term policies (evaluating HDFC Life, Max Life, Tata AIA, and ICICI Prudential) supplemented with 2–3 follow-up discovery questions to initiate the conversation.
  * **Dynamic Knowledge Base Upload**: Sidebar uploader enabling users to index new PDF policy brochures directly into Qdrant in real time.
  * **Zero PII Exposure**: Completely hides sensitive user profile fields and raw JSON dumps from the frontend display.

* **Conversational Checkpointing (`SqliteSaver`)**:
  * Backed by a local `conversations.db` SQLite store mapped to persistent `session_id` threads, allowing users to resume prior advisory conversations without losing extracted context.

* **LangSmith Observability & Benchmarks (`tests/evaluate_rag.py`)**:
  * Configured automated run tracing via LangSmith for node latency, token consumption, and retrieval inspector logs.
  * Built an automated benchmark pipeline using LLM-as-a-judge to evaluate faithfulness, context groundedness, and hallucination rates sequentially against ground-truth policy QA pairs.