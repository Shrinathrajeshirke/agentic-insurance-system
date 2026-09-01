## Defines policy chunk structure and metadata for Qdrant payload filtering

from pydantic import BaseModel, Field
from typing import Optional, List

class PolicyMetadata(BaseModel):
    policy_name: str = Field(..., description="Name of the insurance product")
    insurer: str = Field(..., description="Insurance company name")
    clause_type: str = Field(..., description="e.g., eligibility, exclusion, rider, benefit, claim")
    entry_age_min: int = Field(default=18, description="Minimum allowed entry age")
    entry_age_max: int = Field(default=65, description="Maximum allowed entry age")
    smoker_allowed: bool = Field(default=True, description="Whether plan accepts smokers")
    riders_available: List[str] = Field(default_factory=list, description="List of optional riders")

class PolicyChunk(BaseModel):
    chunk_id: str
    content: str
    metadata: PolicyMetadata