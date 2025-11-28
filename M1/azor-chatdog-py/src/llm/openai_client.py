"""
OpenAI LLM Client Implementation
Encapsulates all OpenAI API interactions.
"""

import os
import sys
from typing import Optional, List, Any, Dict
import openai
from dotenv import load_dotenv
from cli import console
# from .openai_validation import OpenAIConfig  # If you want to use Pydantic validation, mirror GeminiConfig

class OpenAIChatSessionWrapper:
	"""
	Wrapper for OpenAI chat session that provides universal dictionary-based history format.
	Ensures compatibility with Gemini/LlamaClient's history format.
	"""
	def __init__(self, model_name: str, api_key: str, system_instruction: str, history: Optional[List[Dict]] = None):
		self.model_name = model_name
		self.api_key = api_key
		self.system_instruction = system_instruction
		self.history = history or []
		self.messages = self._build_messages()

	def _build_messages(self) -> List[Dict]:
		messages = []
		if self.system_instruction:
			messages.append({"role": "system", "content": self.system_instruction})
		for entry in self.history:
			if isinstance(entry, dict) and 'role' in entry and 'parts' in entry:
				text = entry['parts'][0].get('text', '') if entry['parts'] else ''
				if text:
					messages.append({"role": entry['role'], "content": text})
		return messages

	def send_message(self, text: str) -> Any:
		"""
		Sends a message to OpenAI chat completion endpoint.
		"""
		openai.api_key = self.api_key
		self.messages.append({"role": "user", "content": text})
		try:
			# response = openai.ChatCompletion.create(
			# 	model=self.model_name,
			# 	messages=self.messages
			# )
			response = openai.chat.completions.create(
        model=self.model_name,
        messages=self.messages
      )
			reply = response.choices[0].message["content"]
			self.messages.append({"role": "assistant", "content": reply})
			return reply
		except Exception as e:
			console.print_error(f"Błąd OpenAI: {e}")
			return None

	def get_history(self) -> List[Dict]:
		"""
		Gets conversation history in universal dictionary format.
		"""
		universal_history = []
		for msg in self.messages:
			universal_history.append({
				"role": msg["role"],
				"parts": [{"text": msg["content"]}]
			})
		return universal_history

class OpenAILLMClient:
	"""
	Encapsulates all OpenAI API interactions.
	Provides a clean interface for chat sessions, token counting, and configuration.
	"""
	def __init__(self, model_name: str, api_key: str):
		if not api_key:
			raise ValueError("API key cannot be empty or None")
		self.model_name = model_name
		self.api_key = api_key

	@staticmethod
	def preparing_for_use_message() -> str:
		return "🤖 Przygotowywanie klienta OpenAI..."

	@classmethod
	def from_environment(cls) -> 'OpenAILLMClient':
		load_dotenv()
		model_name = os.getenv('OPENAI_MODEL_NAME', 'gpt-3.5-turbo')
		api_key = os.getenv('OPENAI_API_KEY', '')
		# If you want to use Pydantic validation, mirror GeminiConfig
		# config = OpenAIConfig(model_name=model_name, openai_api_key=api_key)
		# return cls(model_name=config.model_name, api_key=config.openai_api_key)
		return cls(model_name=model_name, api_key=api_key)

	def create_chat_session(self, system_instruction: str, history: Optional[List[Dict]] = None) -> OpenAIChatSessionWrapper:
		return OpenAIChatSessionWrapper(
			model_name=self.model_name,
			api_key=self.api_key,
			system_instruction=system_instruction,
			history=history
		)

	def count_history_tokens(self, history: List[Dict]) -> int:
		"""
		Counts tokens for the given conversation history using tiktoken (if available).
		"""
		try:
			import tiktoken
			enc = tiktoken.encoding_for_model(self.model_name)
			total_tokens = 0
			for entry in history:
				if isinstance(entry, dict) and 'role' in entry and 'parts' in entry:
					text = entry['parts'][0].get('text', '') if entry['parts'] else ''
					if text:
						total_tokens += len(enc.encode(text))
			return total_tokens
		except ImportError:
			console.print_error("tiktoken not installed, cannot count tokens.")
			return 0
		except Exception as e:
			console.print_error(f"Błąd podczas liczenia tokenów: {e}")
			return 0

	def get_model_name(self) -> str:
		return self.model_name

	def is_available(self) -> bool:
		return bool(self.api_key)

	def ready_for_use_message(self) -> str:
		if len(self.api_key) <= 8:
			masked_key = "****"
		else:
			masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}"
		return f"✅ Klient OpenAI gotowy do użycia (Model: {self.model_name}, Key: {masked_key})"
