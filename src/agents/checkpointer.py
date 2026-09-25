import os
from contextlib import contextmanager, asynccontextmanager
from src.config import settings
from src.logger import logger

@asynccontextmanager
async def get_async_checkpointer():
    """
    Yields an asynchronous persistent checkpointer.
    Uses AsyncPostgresSaver (Neon/Supabase) in production when DATABASE_URL is set,
    falling back to AsyncSqliteSaver for local development.
    """
    db_url = settings.DATABASE_URL or os.getenv("DATABASE_URL")

    if db_url and db_url.startswith("postgres"):
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            logger.info("Initializing production AsyncPostgresSaver checkpointer...")
            async with AsyncPostgresSaver.from_conn_string(db_url) as checkpointer:
                await checkpointer.setup()
                yield checkpointer
                return
        except Exception as e:
            logger.warning(f"AsyncPostgresSaver connection failed: {e}. Falling back to SQLite.")

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    logger.info("Using local AsyncSqliteSaver checkpoint fallback.")
    db_path = os.path.join(os.getcwd(), "checkpoints.db")
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        yield checkpointer


@contextmanager
def get_checkpointer():
    """
    Yields a synchronous persistent checkpointer for CLI tools and tests.
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
            logger.warning(f"PostgresSaver connection failed: {e}. Falling back to SQLite.")

    from langgraph.checkpoint.sqlite import SqliteSaver
    logger.info("Using local SqliteSaver checkpoint fallback.")
    db_path = os.path.join(os.getcwd(), "checkpoints.db")
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        yield checkpointer