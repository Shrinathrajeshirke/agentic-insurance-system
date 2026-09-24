# Agentic RAG Insurance Advisory System

An intelligent, stateful term life insurance recommendation and decision-support engine built with LangGraph, Qdrant, FastAPI, and Streamlit.

## Setup & Installation

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Step 1: Base Utilities Setup

To ensure production readiness, custom logging and exception handling modules have been implemented.

- **`src/logger.py`**: Configures dual-logging (console and file-based). Logs are generated dynamically with timestamps in a local `logs/` directory, capturing line numbers, log levels, and execution flow.
- **`src/exception.py`**: Implements a `CustomException` class utilizing Python's `sys` module. It intercepts errors to extract the exact filename, line number, and error description, automatically writing the trace to our logger before halting execution.

## Phase 1: Infrastructure & Environment Setup

- **Vector DB Infrastructure (`docker-compose.yml`)**: Deployed Qdrant with persistent local volume storage (`./data/qdrant_storage`) exposed on ports 6333 (REST) and 6334 (gRPC).
- **Environment & Settings Management (`src/config.py` & `.env`)**: Implemented type-safe environment configuration with `pydantic-settings` configured for OpenAI (`gpt-4o-mini` for agent nodes, `gpt-4o` for evaluation judges), JWT authentication parameters, and local CPU-efficient embeddings (`sentence-transformers/all-MiniLM-L6-v2`).

## Step 2: Domain Schemas, LLM Factory & Verification

- **Domain Schemas (`src/schema/`)**:
  - `user_profile.py`: Pydantic schema for underwriting parameters (Age, Smoker status, Income, Sum Assured, Dependents, Medical history).
  - `policy.py`: Data structure for insurance policy clauses with metadata filtering tags.
  - `state.py`: LangGraph `AgentState` definition tracking messages with `add_messages` reducer, user profile extraction state, retrieved policies, and deterministic underwriting alerts.
- **Model & Embedding Adapters (`src/agents/llm.py`)**: Centralized factory initializing OpenAI (`gpt-4o-mini`) via `ChatOpenAI` alongside CPU-optimized embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
- **Self-Test Script (`test_setup.py`)**: End-to-end verification script testing configurations, schema validation, Qdrant connectivity, embedding vector generation, and OpenAI API calls.

## Phase 2: Data Ingestion & Verifiable Document Tracking

- **Base Dataset (`data/raw_policies/sample_policies.json`)**: Created a structured JSON repository of verified insurance clauses (Eligibility, Exclusions, Riders) for major Indian insurers.
- **Page-Level Vector Indexer (`src/agents/tools.py` & `src/ingestion/indexer.py`)**:
  - Ingestion pipeline utilizing `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
  - Dynamic section detection mapping chunk text to contract categories (Exclusions & Restrictions, Grace Period & Revival, Free-Look Period, Riders, Surrender & Exit Value, and Claims).
  - Captures 1-indexed document page numbers and policy contract metadata in Qdrant payloads, enabling verifiable audit trails.
  - Compatible with `qdrant-client` modern `query_points` and fallback `search` APIs.

## Phase 3: Agentic Tools, Guardrails & Actuarial Calculator

- **Deterministic Underwriting Guardrail (`src/schema/guardrails.py`)**:
  - Non-probabilistic rule engine enforcing strict IRDAI Human Life Value (HLV) income multipliers:
    - Age $\le 35$: up to $25\times$ annual income
    - Age $36 - 45$: up to $20\times$ annual income
    - Age $46 - 55$: up to $15\times$ annual income
    - Age $> 55$: up to $10\times$ annual income
  - Evaluates entry age limits (18 to 65 years) and maximum maturity ceilings (85 years), flagging high-cover gating without audited financial documents (ITR-V/Form 16).
- **Actuarial Rate Calculator (`src/tools/calculator.py`)**:
  - Deterministic pricing matrix computing exact annual and monthly base rates across age brackets for the top 4 plans (HDFC Life Click 2 Protect Super, Max Life Smart Secure Plus, Tata AIA Sampoorna Raksha Supreme, ICICI Pru iProtect Smart).
  - Evaluates smoker loading factors, female applicant discounts, optional rider costs (Critical Illness and Accidental Death), and Return of Premium (TROP) multipliers.
  - Enforces the individual term life insurance 0% GST policyholder tax rule.
- **Policy Vector Search (`search_policy_contracts`)**: LangChain tool querying the Qdrant vector index with metadata extraction (page numbers, section classification, similarity scores).
- **Fallback Web Retrieval (`web_search`)**: DuckDuckGo regional search (`in-en`) with grounded benchmark fallbacks for Indian market rates and Claim Settlement Ratios when external requests are throttled.

## Phase 4: Stateful Agent Orchestration (LangGraph)

- **`src/agents/graph.py`**: Asynchronous `StateGraph` workflow:
  - `profiler_node`: Asynchronously extracts and updates structured applicant metadata (`UserProfile`) across conversational turns.
  - `guardrail_node`: Evaluates profile parameters through the deterministic underwriting rule engine, injecting non-negotiable underwriting flags before retrieval.
  - `retriever_node`: Executes exact demographic rate calculations, searches Qdrant for contract evidence, and conditionally enriches context with benchmark market disclosures.
  - `advisor_node`: Synthesizes verified contract excerpts, calculated rupee numbers, and underwriting alerts into an IRDAI-compliant advisory assessment with cited reference tags.
- **Graph Compilation**: Fully asynchronous workflow compiled with `AsyncSqliteSaver` checkpointer integration for persistent conversation sessions.

## Phase 5: Privacy, Persistence, Reporting & UI

- **Data Privacy & Sanitization (`src/security/sanitizer.py`)**:
  - Pure-Python regex pattern matching to mask structured identifiers (Credit Cards, SSNs, Emails, Phone numbers, Dates of Birth, ZIP codes).
  - Dictionary key scrubbing to mask sensitive user profile traits (Medical conditions, Gender) before writing state to logs or internal databases.
- **FastAPI Service Layer (`src/api/server.py`)**:
  - Built-in lifespan management initializing `AsyncSqliteSaver` against `conversations.db`.
  - Secure JWT authentication endpoints (`/auth/register`, `/auth/login`).
  - Thread lifecycle management routes to create, list, rename (`PATCH /threads/{id}/rename`), and delete (`DELETE /threads/{id}`) advisory sessions.
  - Real-time Server-Sent Events (SSE) streaming endpoint (`/chat/stream`) emitting incremental token chunks and citation metadata payloads.
  - Multipart document ingestion endpoint (`/upload_brochure`) indexing custom PDF filings with page-level tracking.
  - Report export endpoint (`/export_advisory_pdf`) serving generated PDF advisory briefs.
- **Automated PDF Advisory Report Generator (`src/tools/report_generator.py`)**:
  - Built with ReportLab to compile structured advisory briefs on demand.
  - Synthesizes applicant underwriting data, side-by-side benchmark comparison tables, sanitized advisory recommendations, and statutory IRDAI notices into downloadable PDF documents.
- **Streamlit Web Application (`app.py`)**:
  - **Authentication Views**: Tabbed login and registration handling JWT bearer tokens.
  - **Sidebar Thread Management**: Allows switching between previous advisory chats, renaming threads inline, and deleting inactive sessions with instant state resets.
  - **Underwriting Assessment Wizard**: Collects age, gender, smoker habits, dependents, salary brackets, and sum assured before unlocking the chat session.
  - **Interactive Proof Accordions**: Expandable "🔍 Verified Contract Evidence & Citations" container under every assistant turn displaying document names, exact page numbers, section headers, and quoted clauses.
  - **Side-by-Side Comparison Matrix**: Expandable table comparing Claim Settlement Ratio (CSR by count), Amount Settlement Ratio (ASR by value), Critical Illness counts, Zero-Cost Exit (SEV) options, and suicide exclusion rules across top insurers.
  - **PDF Download Action**: One-click "📥 Download PDF Report" button retrieving compiled ReportLab advisory briefs from the backend.
  - **Modern Layout Standards**: Compliant with updated Streamlit container specifications (`width="stretch"` and `width="content"`).
- **Conversational Checkpointing (`AsyncSqliteSaver`)**:
  - Local `conversations.db` SQLite store mapped to persistent `thread_id` records, preserving chat transcripts and extracted user state across restarts.