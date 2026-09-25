import os
import sqlite3
from contextlib import contextmanager
from src.config import settings
from src.logger import logger

@contextmanager
def get_checkpointer():
    """
    Yields a persistent checkpointer.
    Uses managed PostgresSaver (Neon/Supabase) in production when DATABASE_URL is set,
    falling back to SQLite for local development.
    """
    db_url = settings.DATABASE_URL or os.getenv("DATABASE_URL")
    
    if db_url and db_url.startswith("postgres"):
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            logger.info("Initializing production PostgresSaver checkpointer...")
            with PostgresSaver.from_conn_string(db_url) as checkpointer:
                checkpointer.setup()
                yield checkpointer
                return
        except Exception as e:
            logger.warning(f"Postgres connection failed: {e}. Falling back to SQLite.")

    # Local fallback: use plain file path or in-memory sqlite
    from langgraph.checkpoint.sqlite import SqliteSaver
    logger.info("Using local SqliteSaver checkpoint fallback.")
    db_path = os.path.join(os.getcwd(), "checkpoints.db")
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        yield checkpointer