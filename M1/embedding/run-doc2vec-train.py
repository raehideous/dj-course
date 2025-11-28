import json
import logging
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tokenizers import Tokenizer
import time
from corpora import CORPORA_FILES

# Ustawienie logowania dla gensim
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

# Parametry treningu Doc2Vec
VECTOR_LENGTH = 200
WINDOW_SIZE = 5
MIN_COUNT = 3
WORKERS = 4
EPOCHS = 400
SG_MODE = 0


def train_and_save_doc2vec_model(tagged_data, output_model_file):
    print("\n--- Rozpoczynanie Treningu Doc2Vec ---")
    start_time = time.time()
    # 1. Inicjalizacja: Właściwa inicjalizacja, BEZ podawania tagged_data
    model_d2v = Doc2Vec(
        tagged_data,
        vector_size=VECTOR_LENGTH,
        window=WINDOW_SIZE,
        min_count=MIN_COUNT,
        workers=WORKERS,
        epochs=EPOCHS,
    )

    end_time = time.time()
    print("\nTrening zakończony pomyślnie.")

    print(f"Czas treningu: {end_time - start_time:.2f} sekund.")
    print(
        f"Parametrów modelu Doc2Vec: vector_size={VECTOR_LENGTH}, window={
            WINDOW_SIZE
        }, min_count={MIN_COUNT}, epochs={EPOCHS}"
    )


    try:
        model_d2v.save(output_model_file)
        print(f"\nPełny model Doc2Vec zapisany jako: '{output_model_file}'.")

    except Exception as e:
        # W kontekście 'połączonego skryptu' błąd zapisu nie przerywa wnioskowania
        print(
            f"OSTRZEŻENIE: BŁĄD podczas zapisu modelu/mapy: {
                e
            }"
        )

    return model_d2v


EMBEDDINGS_DIR = "./embeddings/"

FILES = {
    "WOLNELEKTURY": CORPORA_FILES["WOLNELEKTURY"],
    "ALL": CORPORA_FILES["ALL"],
    "PAN_TADEUSZ": CORPORA_FILES["PAN_TADEUSZ"],
}
file_name = "ALL"
files = FILES[file_name]

TOKENIZERS = {
    "all-corpora": "../tokenizer/tokenizers/all-corpora_32k.json",
    "wolne-lektury": "../tokenizer/tokenizers/wolne-lektury_32k.json",
    "pan-tadeusz": "../tokenizer/tokenizers/pan-tadeusz_32k.json",
    "deepseek-r1": "../tokenizer/tokenizers/deepseek-r1.json",
}

tokenizer_name = "all-corpora"
tokenizer = Tokenizer.from_file(TOKENIZERS[tokenizer_name])

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
    except Exception as e:
        print(f"BŁĄD podczas przetwarzania pliku '{file}': {e}")
        continue

if not raw_sentences:
    print("BŁĄD: Korpus danych jest pusty.")
    raise ValueError("Korpus danych jest pusty.")

# Konwersja na listę tokenów
tokenized_sentences = [tokenizer.encode(sentence).tokens for sentence in raw_sentences]

# Przygotowanie danych dla Doc2Vec
tagged_data = [
    TaggedDocument(words=tokenized_sentences[i], tags=[str(i)])
    for i in range(len(tokenized_sentences))
]
print(f"Przygotowano {len(tagged_data)} sekwencji TaggedDocument do treningu.")

model_path = f"{EMBEDDINGS_DIR}{file_name}_doc2vec_model_{tokenizer_name}.model"
sentence_map_path = (
    f"{EMBEDDINGS_DIR}{file_name}_doc2vec_sentence_map_{tokenizer_name}.json"
)

with open(sentence_map_path, "w", encoding="utf-8") as f:
    json.dump(raw_sentences, f, ensure_ascii=False, indent=4)
print(f"Mapa zdań do ID zapisana jako: '{sentence_map_path}'.")

model_d2v = train_and_save_doc2vec_model(tagged_data, model_path)

# 1. Sprawdzenie, czy wektor dla Tag_4 jest najbardziej podobny do siebie (NATYCHMIAST PO TRENINGU)
tag_do_sprawdzenia = '4'

if tag_do_sprawdzenia in model_d2v.dv:
    # Wyszukiwanie podobnych dokumentów do WYTRENOWANEGO WEKTORA
    similar_docs = model_d2v.dv.most_similar([tag_do_sprawdzenia])

    print("\n" + "=" * 50)
    print(f"DIAGNOSTYKA: WYNIK NATYCHMIAST PO TRENINGU DLA TAGU '{tag_do_sprawdzenia}':")
    for tag, similarity in similar_docs:
        print(f"- Tag: {tag}, Similarity: {similarity:.10f}")
else:
    print(f"BŁĄD KRYTYCZNY: Tag '{tag_do_sprawdzenia}' nie istnieje w Doc Vectors.")
