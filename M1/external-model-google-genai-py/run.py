from google import genai
import tiktoken
from google.genai import types
import os
import mlflow

from dotenv import load_dotenv
load_dotenv()

# if set, print first 4 chars and last 4 chars and dots inside, else print NOT SET
print(f"env var \"GEMINI_API_KEY\" is:{ os.getenv('GEMINI_API_KEY', '')[:4] + '...' + os.getenv('GEMINI_API_KEY', '')[-4:] if len(os.getenv('GEMINI_API_KEY', '')) > 0 else 'NOT SET' }")
if not os.getenv('GEMINI_API_KEY'):
    raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it to your Google Gemini API key.")

mlflow.gemini.autolog()
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Gemini")


client = genai.Client()

model = "gemini-2.5-flash"
# model = "gemini-1.5-pro"

system_role = "you were Gandalf the Grey in the Lord of the Rings. You answer in max 15 words. Your answers are mysterious and magical."

conversation_history = [
    types.Content(
        role="user",
        parts=[types.Part.from_text(text="What is the best time for coffee?")]
    ),
    types.Content(
        role="model",
        parts=[types.Part.from_text(text="The best time for coffee is in the morning my apprentice.")]
    ),
    types.Content(
        role="user",
        parts=[types.Part.from_text(text="How about tea?")]
    ),
]

response = client.models.generate_content(
    model=model,
    contents=conversation_history,
    config=types.GenerateContentConfig(
        system_instruction=system_role,
        thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
    ),
)

print(response.text)

# Get the trace object just created
trace = mlflow.get_trace(mlflow.get_last_active_trace_id())

enc = tiktoken.get_encoding("o200k_base")
tokens = enc.encode(response.text)
print(tokens)
