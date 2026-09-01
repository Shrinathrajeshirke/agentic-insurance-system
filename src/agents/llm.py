import sys
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import settings
from src.logger import logger
from src.exception import CustomException
import os

def get_llm():
    """Initializes the Groq LLM instance using config parameters."""
    try:
        if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "your_actual_groq_api_key_here":
            raise ValueError("GROQ_API_KEY is not configured in .env")

        logger.info(f"Connecting to Groq LLM: {settings.GROQ_MODEL_NAME}")
        return ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.GROQ_MODEL_NAME,
            temperature=0.1
        )
    except Exception as e:
        raise CustomException(e, sys)

def get_embeddings():
    """Initializes local CPU embeddings (all-MiniLM-L6-v2)."""
    try:
        logger.info(f"Loading local embedding model: {settings.EMBEDDING_MODEL_NAME}")
        return HuggingFaceEmbeddings(
            model_name = settings.EMBEDDING_MODEL_NAME,
            model_kwargs={"device":"cpu"}
        )
    except Exception as e:
        raise CustomException(e, sys)
    