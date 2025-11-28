from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
import os

class OpenAIConfig(BaseModel):
    engine: Literal["OPENAI"] = Field(default="OPENAI")
    openai_api_key: str = Field(..., min_length=1, description="Klucz API OpenAI")
      
    @validator('openai_api_key')
    def validate_api_key(cls, v):
        if not v or v.strip() == "":
            raise ValueError("OPENAI_API_KEY nie może być pusty")
        return v.strip()
