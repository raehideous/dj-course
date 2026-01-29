import time
import threading
from TTS.api import TTS
import warnings 
import os
from cli import console
from files.config import LOG_DIR


warnings.filterwarnings("ignore", category=UserWarning)

FILE_PATH = os.path.join(os.path.dirname(__file__), "sample-agent.wav")
OUTPUT_WAV_PATHS = "output.wav"
AUDIO_DIR = os.path.join(LOG_DIR, 'output_audio')

_tts_instance = None
_tts_lock = threading.Lock()

def get_tts_instance():
    global _tts_instance
    if _tts_instance is None:
        with _tts_lock:
            if _tts_instance is None:
                _tts_instance = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
    return _tts_instance

def speak(parts: list, session_id: str, language: str = "pl", speaker_wav: str = FILE_PATH) -> str:
    """
    Accepts a list of message parts (each a dict with 'text' key) and reads them aloud using TTS.
    Returns the path to the generated audio file.
    Example input: parts=[{"text": "Hello"}, {"text": "world!"}]
    """
    if not parts or not isinstance(parts, list):
        console.print_error("No valid message parts to synthesize.")
        return ""
    text = " ".join(part.get('text', '') for part in parts if isinstance(part, dict) and 'text' in part).strip()
    text = sanitize_text(text)
    if not text:
        console.print_error("No text found in message parts.")
        return ""

    try:
        tts = get_tts_instance()
    except Exception as e:
        console.print_error(f"TTS model load failed: {e}")
        return ""

    # Output directory and filename
    os.makedirs(AUDIO_DIR, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_wav_path = os.path.join(AUDIO_DIR, f"tts_{session_id}_{timestamp}.wav")
    # Synthesize
    tts.tts_to_file(
        text=text,
        file_path=output_wav_path,
        speaker_wav=speaker_wav,
        language=language
    )
    console.print_info(f"Audio generated: {output_wav_path}")
    return output_wav_path

# Sanitize text: remove unreadable characters (non-printable, control, etc.)
def sanitize_text(s):
    import unicodedata
    return ''.join(c for c in s if (unicodedata.category(c)[0] != 'C' and c.isprintable()))

