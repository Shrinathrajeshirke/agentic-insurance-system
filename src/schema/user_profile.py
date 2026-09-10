from pydantic import BaseModel, Field
from typing import List, Optional, Union

class UserProfile(BaseModel):
    age: Optional[int] = Field(default=None, description="Age in years")
    gender: Optional[str] = Field(default=None, description="Gender")
    annual_income: Optional[Union[str, float]] = Field(
        default=None, 
        description="Annual income tier or exact value in INR"
    )
    is_smoker: Optional[bool] = Field(default=None, description="Tobacco/smoking status")
    medical_conditions: List[str] = Field(default_factory=list, description="Declared health conditions")
    dependents_count: Optional[int] = Field(default=None, description="Count of dependents")
    desired_sum_assured: Optional[Union[str, float]] = Field(
        default=None, 
        description="Desired life cover tier (e.g., '1 Crore', '2 Crore')"
    )
    policy_term_years: Optional[int] = Field(default=None, description="Coverage duration in years")