from cli import console
import threading
from audio.speak import speak
from session import ChatSession
from TTS.api import TTS

FILE_PATH = "sample-agent.wav"

def read_last_response(session: ChatSession) -> None:
    """Reads aloud the last assistant response using text-to-speech."""
    history = session.get_history()
    if not history:
        print("Session history is empty. No response to read.")
        return
    
    # Find the last assistant message with 'parts'
    for message in reversed(history):
        console.print_debug(f"Checking message: {message}")
        if message['role'] == 'model' and 'parts' in message:
            parts = message['parts']
            break
    else:
        print("No assistant response with 'parts' found in session history.")
        return

    # Use the new TTS function
    speak(parts, session.session_id)

