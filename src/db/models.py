import os
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from src.config import settings

# 1. Resolve connection string: prioritize PostgreSQL (Neon), fall back to SQLite
raw_db_url = settings.DATABASE_URL or os.getenv("DATABASE_URL") or "sqlite:///./app_data.db"

# SQLAlchemy 1.4+ / 2.0 requires 'postgresql://' instead of legacy 'postgres://'
if raw_db_url.startswith("postgres://"):
    raw_db_url = raw_db_url.replace("postgres://", "postgresql://", 1)

# SQLite requires check_same_thread=False, PostgreSQL does not
engine_kwargs = {"pool_pre_ping": True}
if raw_db_url.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(raw_db_url, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    threads = relationship("ChatThread", back_populates="owner", cascade="all, delete-orphan")


class ChatThread(Base):
    __tablename__ = "chat_threads"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    title = Column(String, default="New Advisory Session")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    owner = relationship("User", back_populates="threads")


def init_db():
    """Idempotently create tables in PostgreSQL or SQLite."""
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()