from pydantic import BaseModel, Field, validator
from typing import Literal

class AnthropicConfig(BaseModel):
    engine: Literal["ANTHROPIC"] = Field(default="ANTHROPIC")
    model_name: str = Field(..., description="Anthropic model name")
    anthropic_api_key: str = Field(..., min_length=1, description="Anthropic API key")

    @validator('anthropic_api_key')
    def validate_api_key(cls, v):
        if not v or v.strip() == "":
            raise ValueError("ANTHROPIC_API_KEY cannot be empty")
        return v.strip()
