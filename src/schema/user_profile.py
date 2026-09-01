## Defines the customer's underwriting profile.

from pydantic import BaseModel, Field
from typing import Optional, List

class UserProfile(BaseModel):
    age: Optional[int] = Field(None, description="Age in years")
    gender: Optional[str] = Field(None, description="Male, Female, or Other")
    annual_income: Optional[float] = Field(None, description="Annual income in local currency")
    is_smoker: Optional[bool] = Field(None, description="True if tobacco/nicotine user, False otherwise")
    medical_conditions: List[str] = Field(default_factory=list, description="List of pre-existing conditions")
    dependents_count: Optional[int] = Field(None, description="Number of financial dependents")
    desired_sum_assured: Optional[float] = Field(None, description="Target cooverage amount")
    policy_term_years: Optional[int] = Field(None, description="Desired coverage duration in years")
