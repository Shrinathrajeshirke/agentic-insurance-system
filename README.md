# Agentic RAG Insurance Advisory System

An intelligent, stateful term life insurance recommendation engine built with LangGraph, Qdrant, Ollama, and FastAPI.

## Step 1: Base Utilities Setup
To ensure production readiness, custom logging and exception handling modules have been implemented.

* **`src/logger.py`**: Configures dual-logging (console and file-based). Logs are generated dynamically with timestamps in a local `logs/` directory, capturing line numbers, log levels, and execution flow.
* **`src/exception.py`**: Implements a `CustomException` class utilizing Python's `sys` module. It intercepts errors to extract the exact filename, line number, and error description, automatically writing the trace to our logger before halting execution.