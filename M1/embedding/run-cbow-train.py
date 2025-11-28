import numpy as np
import json
import logging
from gensim.models import Word2Vec
from tokenizers import Tokenizer

from corpora import CORPORA_FILES  # type: ignore

# Ustawienie logowania dla gensim
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


# loading r& aggregating aw sentences from files
def aggregate_raw_sentences(files):
    raw_sentences = []
    print("Wczytywanie tekstu z plików...")
    print(f"Liczba plików do wczytania: {len(files)}")
    for file in files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                raw_sentences.extend(lines)
        except FileNotFoundError:
            print(f"OSTRZEŻENIE: Nie znaleziono pliku '{file}'. Pomijam.")
            continue

    if not raw_sentences:
        print("BŁĄD: Pliki wejściowe są puste lub nie zostały wczytane.")
        exit()
    return raw_sentences


# Parametry treningu Word2Vec (CBOW)
VECTOR_LENGTH = 20
WINDOW_SIZE = 6
MIN_COUNT = 2
WORKERS = 4
EPOCHS = 20
SAMPLE_RATE = 1e-2
SG_MODE = 0  # 0 dla CBOW, 1 dla Skip-gram


def train_and_save_word2vec_model(
    tokenizer_name: str,
    tokenizer: Tokenizer,
    raw_sentences,
    output_tensor_file: str,
    output_map_file: str,
    output_model_file: str,
):
    print(f"Tokenizacja {len(raw_sentences)} zdań...")
    encodings = tokenizer.encode_batch(raw_sentences)

    # Konwersja obiektów Encoding na listę list stringów (tokenów)
    tokenized_sentences = [encoding.tokens for encoding in encodings]
    print(f"Przygotowano {len(tokenized_sentences)} sekwencji do treningu.")

    # --- ETAP 2: Trening Word2Vec (CBOW) ---

    print("\n--- Rozpoczynanie Treningu Word2Vec (CBOW) ---")
    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=VECTOR_LENGTH,
        window=WINDOW_SIZE,
        min_count=MIN_COUNT,
        workers=WORKERS,
        sg=SG_MODE,  # 0: CBOW
        epochs=EPOCHS,
        sample=SAMPLE_RATE,
    )
    print("Trening zakończony pomyślnie.")

    # --- ETAP 3: Eksport i Zapis Wyników ---

    # Eksport tensora embeddingowego
    embedding_matrix_np = model.wv.vectors
    embedding_matrix_tensor = np.array(embedding_matrix_np, dtype=np.float32)

    print(
        f"\nKształt finalnego tensora: {
            embedding_matrix_tensor.shape
        } (Tokeny x Wymiar)"
    )

    # 1. Zapisanie tensora NumPy (.npy)
    np.save(output_tensor_file, embedding_matrix_tensor)
    print(f"Tensor embeddingowy zapisany jako: '{output_tensor_file}'.")

    # 2. Zapisanie mapowania tokenów na indeksy
    token_to_index = {
        token: model.wv.get_index(token) for token in model.wv.index_to_key
    }
    with open(output_map_file, "w", encoding="utf-8") as f:
        json.dump(token_to_index, f, ensure_ascii=False, indent=4)
    print(f"Mapa tokenów do indeksów zapisana jako: '{output_map_file}'.")

    # 3. Zapisanie całego modelu gensim (opcjonalne, ale zalecane)
    model.save(output_model_file)
    print(f"Pełny model Word2Vec zapisany jako: '{output_model_file}'.")


EMBEDDINGS_DIR = "./embeddings/cbow/"

FILES = {
    "WOLNELEKTURY": CORPORA_FILES["WOLNELEKTURY"],
    "ALL": CORPORA_FILES["ALL"],
    "PAN_TADEUSZ": CORPORA_FILES["PAN_TADEUSZ"],
}

TOKENIZERS = {
    "all-corpora": "../tokenizer/tokenizers/all-corpora_32k.json",
    "wolne-lektury": "../tokenizer/tokenizers/wolne-lektury_32k.json",
    "pan-tadeusz": "../tokenizer/tokenizers/pan-tadeusz_32k.json",
    "deepseek-r1": "../tokenizer/tokenizers/deepseek-r1.json",
}

for tokenizer_name, tokenizer_path in TOKENIZERS.items():
    if tokenizer_name != "all-corpora":
        continue  # Uruchamiamy tylko dla tokenizera "pan-tadeusz" jako przykład
    print(f"\n=== Przetwarzanie z tokenizatorem: {tokenizer_name} ===")

    # --- ETAP 1: Wczytanie i Tokenizacja Korpusu ---
    try:
        print(f"Ładowanie tokenizera z pliku: {tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path)
    except FileNotFoundError:
        print(
            f"BŁĄD: Nie znaleziono pliku '{
                tokenizer_path
            }'. Upewnij się, że plik istnieje."
        )
        raise

    for file_name, file in FILES.items():
        raw_sentences = aggregate_raw_sentences(file)

        tensor_path = f"{EMBEDDINGS_DIR}{file_name}_tensor_{tokenizer_name}_cbow.npy"
        map_path = f"{EMBEDDINGS_DIR}{file_name}_map_{tokenizer_name}_cbow.json"
        model_path = f"{EMBEDDINGS_DIR}{file_name}_model_{tokenizer_name}_cbow.model"

        train_and_save_word2vec_model(
            tokenizer_name, tokenizer, raw_sentences, tensor_path, map_path, model_path
        )
