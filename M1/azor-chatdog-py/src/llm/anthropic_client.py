"""
Anthropic Claude LLM Client Implementation
Encapsulates all Anthropic Claude AI interactions.
"""

import os
import sys
from typing import Optional, List, Any, Dict
from anthropic import Anthropic
from dotenv import load_dotenv
from cli import console
from .anthropic_validation import AnthropicConfig

class AnthropicChatSessionWrapper:
    """
    Wrapper for Anthropic chat session that provides universal dictionary-based history format.
    Ensures compatibility with Gemini/LlamaClient's history format.
    """
    def __init__(self, anthropic_client, model_name, system_instruction, temperature, top_p, top_k, history=None):
        self.anthropic_client = anthropic_client
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.history = history or []
        self.messages = []
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def send_message(self, text: str) -> Any:
        # Build message history for Anthropic
        messages = []
        if self.history:
            for entry in self.history:
                if isinstance(entry, dict) and 'role' in entry and 'parts' in entry:
                    role = entry['role']
                    content = entry['parts'][0].get('text', '') if entry['parts'] else ''
                    if role == 'user':
                        messages.append({"role": "user", "content": content})
                    elif role == 'model':
                        messages.append({"role": "assistant", "content": content})
        # Add current user message
        messages.append({"role": "user", "content": text})
        response = self.anthropic_client.messages.create(
            model=self.model_name,
            system=self.system_instruction,
            messages=messages,
            max_tokens=1024,
            temperature=self.temperature,
            # top_p=self.top_p, # Cannot be used when temperature is set
            top_k=self.top_k
        )
        self.messages.append(response)
        return AnthropicResponse(response)

    def get_history(self) -> List[Dict]:
        # Convert Anthropic messages to universal format, ensuring JSON serializability
        universal_history = []
        for msg in self.messages:
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                # msg.content may be a list of TextBlock objects or a string
                if isinstance(msg.content, list) and len(msg.content) > 0:
                    # If it's a list, extract text from each block
                    parts = []
                    for part in msg.content:
                        if isinstance(part, dict):
                            text = part.get('text', str(part))
                        elif hasattr(part, 'text'):
                            text = part.text
                        else:
                            text = str(part)
                        parts.append({"text": text})
                else:
                    # If it's a string or other type
                    text = msg.content if isinstance(msg.content, str) else str(msg.content)
                    parts = [{"text": text}]
                universal_history.append({
                    "role": "model" if msg.role == "assistant" else "user",
                    "parts": parts
                })
        return universal_history

class AnthropicLLMClient:
    """
    Encapsulates all Anthropic Claude AI interactions.
    Provides a clean interface for chat sessions, token counting, and configuration.
    """
    def __init__(self, model_name: str, api_key: str):
        if not api_key:
            raise ValueError("API key cannot be empty or None")
        self.model_name = model_name
        self.api_key = api_key
        self._client = self._initialize_client()

    @staticmethod
    def preparing_for_use_message() -> str:
        return "🤖 Przygotowywanie klienta Anthropic..."

    @classmethod
    def from_environment(cls) -> 'AnthropicLLMClient':
        load_dotenv()
        config = AnthropicConfig(
            model_name=os.getenv('MODEL_NAME', 'claude-sonnet-4-5'),
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY', '')
        )
        return cls(model_name=config.model_name, api_key=config.anthropic_api_key)

    def _initialize_client(self) -> Anthropic:
        try:
            return Anthropic(api_key=self.api_key)
        except Exception as e:
            console.print_error(f"Błąd inicjalizacji klienta Anthropic: {e}")
            sys.exit(1)

    def create_chat_session(
                self,
                system_instruction: str,
                history: Optional[List[Dict]] = None,
                temperature: float = 1.0,
                top_p: float = 1.0,
                top_k: int = 40
            ) -> AnthropicChatSessionWrapper:
        if not self._client:
            raise RuntimeError("LLM client not initialized")
        return AnthropicChatSessionWrapper(
            anthropic_client=self._client,
            model_name=self.model_name,
            system_instruction=system_instruction,
            history=history,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )

    def count_history_tokens(self, history: List[Dict]) -> int:
        # Anthropic does not expose a direct token counting API, so we estimate
        if not history:
            return 0
        try:
            # Simple estimation: 1 token ≈ 4 characters (English)
            total_chars = 0
            for entry in history:
                if isinstance(entry, dict) and 'parts' in entry:
                    part = entry['parts'][0] if entry['parts'] else ''
                    # Handle dicts and TextBlock objects
                    if isinstance(part, dict):
                        text = part.get('text', '')
                    elif hasattr(part, 'text'):
                        text = part.text
                    else:
                        text = str(part)
                    total_chars += len(text)
            return total_chars // 4
        except Exception as e:
            console.print_error(f"Błąd podczas szacowania tokenów: {e}")
            return 0

    def get_model_name(self) -> str:
        return self.model_name

    def is_available(self) -> bool:
        return self._client is not None and bool(self.api_key)

    def ready_for_use_message(self) -> str:
        if len(self.api_key) <= 8:
            masked_key = "****"
        else:
            masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}"
        return f"✅ Klient Anthropic gotowy do użycia (Model: {self.model_name}, Key: {masked_key})"

    @property
    def client(self):
        return self._client

class AnthropicResponse:
    """
    Represents a response from the Anthropic LLM, extracting only the text content.
    """
    def __init__(self, response):
        # Anthropic's response content is a list of TextBlock objects
        # Extract the first text block's text
        if hasattr(response, 'content') and response.content:
            # If content is a list of TextBlock, get the first one's text
            self.text = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
        else:
            self.text = str(response)
    def __str__(self):
        return self.text
