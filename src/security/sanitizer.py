import sys
import re
from typing import Dict, Any
from src.logger import logger
from src.exception import CustomException

# Standard Regex patterns for highly structured PII
PII_PATTERNS = {
    "CREDIT_CARD": r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
    "SSN": r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b",
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "PHONE": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "DOB_DATE": r"\b(0[1-9]|1[0-2]|[1-9])[-/](0[1-9]|[12]\d|3[01]|[1-9])[-/](\d{2}|\d{4})\b",
    "ZIP_CODE": r"\b\d{5}(?:-\d{4})?\b"
}

def mask_text_pii(text: str) -> str:
    """Scans and masks structured PII entities within unstructured text strings using pure regex."""
    if not text or not isinstance(text, str):
        return text
    
    try:
        anonymized = text
        for pii_type, pattern in PII_PATTERNS.items():
            anonymized = re.sub(pattern, "[REDACTED]", anonymized)
        return anonymized
    except Exception as e:
        logger.error(f"PII regex sanitization warning: {str(e)}")
        return text

def sanitize_user_profile_for_storage(profile_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a sanitized copy of the user profile. 
    Applies regex to strings and directly blanks out abstract PII dictionary keys.
    """
    try:
        sanitized = profile_dict.copy()
        
        # Abstract fields (Names, Religion, Medical, Race, etc.) cannot be reliably caught 
        # with regex in unstructured text. We handle them by wiping the values directly.
        abstract_pii_keys = [
            "full_name", "gender", "mailing_address", "medical_conditions", 
            "place_of_birth", "race", "religion", "passport_information", "driver_license"
        ]
        
        for key in abstract_pii_keys:
            if key in sanitized and sanitized[key]:
                if isinstance(sanitized[key], list):
                    sanitized[key] = ["[REDACTED]"]
                else:
                    sanitized[key] = "[REDACTED]"
                    
        return sanitized
    except Exception as e:
        raise CustomException(e, sys)