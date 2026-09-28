import pytest
from unittest.mock import AsyncMock, MagicMock
from contextlib import asynccontextmanager
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from langchain_core.messages import AIMessage

from src.api.server import app
import src.api.server as server_module
from src.db.models import Base, get_db, User, ChatThread
from src.security.auth import get_current_user

# 1. Shared In-Memory SQLite database via StaticPool
engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=engine)

# Create all tables on the shared pool
Base.metadata.create_all(bind=engine)

# Seed test user
test_user = User(id="test-user-uuid", email="advisor_test@example.com", hashed_password="pw")
with TestingSessionLocal() as session:
    session.add(test_user)
    session.commit()

def override_get_current_user():
    return test_user

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

# 2. No-op Lifespan so tests bypass remote Postgres checkpointer entirely
@asynccontextmanager
async def mock_lifespan(fastapi_app):
    yield


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.ainvoke = AsyncMock(return_value={
        "messages": [AIMessage(content="IRDAI advises pure term life coverage up to 25x annual income.")],
        "user_profile": {"age": 28},
        "underwriting_flags": [],
        "rate_quotes": [{"plan_name": "HDFC Life", "annual_premium": 11200}]
    })
    return agent


@pytest.fixture
def client(mock_agent):
    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_db] = override_get_db

    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = mock_lifespan
    server_module.agent_app = mock_agent

    with TestClient(app, raise_server_exceptions=True) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    app.router.lifespan_context = original_lifespan
    server_module.agent_app = None


def test_health_docs(client):
    response = client.get("/docs")
    assert response.status_code == 200


def test_sync_chat_endpoint(client):
    payload = {
        "thread_id": "test-thread-001",
        "messages": [{"role": "user", "content": "How much cover can I get?"}],
        "user_profile": {"age": 28, "annual_income": "10 - 15 Lakhs"}
    }

    response = client.post("/chat", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
    data = response.json()
    assert "IRDAI advises" in data["reply"]
    assert len(data["quotes"]) == 1
    assert data["quotes"][0]["plan_name"] == "HDFC Life"


def test_unauthenticated_chat_rejected():
    app.dependency_overrides.clear()
    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = mock_lifespan

    with TestClient(app, raise_server_exceptions=False) as unauth_client:
        response = unauth_client.post("/chat", json={
            "thread_id": "thread-unauth",
            "messages": [{"role": "user", "content": "Hello"}]
        })
        assert response.status_code == 401

    app.router.lifespan_context = original_lifespan