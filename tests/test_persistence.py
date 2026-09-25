import pytest
from src.config import settings
from src.agents.checkpointer import get_checkpointer

def test_database_checkpointer_connection():
    """Verifies that the checkpointer connects and executes .setup() without error."""
    with get_checkpointer() as checkpointer:
        assert checkpointer is not None
        # Setup creates the required schema tables if they don't exist
        checkpointer.setup()