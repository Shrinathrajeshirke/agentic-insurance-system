from contextlib import contextmanager
from src.config import settings
from src.logger import logger

@contextmanager
def get_checkpointer():
    """
    Yields a persistent checkpointer.
    Uses managed PostgresSaver (Neon/Supabase) in production,
    falling back to SQLite for local development.
    """
    if settings.DATABASE_URL and settings.DATABASE_URL.startswith("postgres"):
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            logger.info("Initializing production PostgresSaver checkpointer...")
            
            # Use connection string factory
            with PostgresSaver.from_conn_string(settings.DATABASE_URL) as checkpointer:
                # Runs one-time table creation migrations if not present
                checkpointer.setup()
                yield checkpointer
                return
        except Exception as e:
            logger.warning(f"Postgres connection failed: {e}. Falling back to SQLite.")

    # Local fallback
    from langgraph.checkpoint.sqlite import SqliteSaver
    logger.info("Using local SqliteSaver checkpoint fallback.")
    with SqliteSaver.from_conn_string("sqlite:///checkpoints.db") as checkpointer:
        yield checkpointer