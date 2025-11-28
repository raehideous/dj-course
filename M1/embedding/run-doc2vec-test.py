from gensim.models.doc2vec import Doc2Vec
from tokenizers import Tokenizer

from corpora import CORPORA_FILES

print("\n" + "=" * 50)

FILES = {
    "WOLNELEKTURY": CORPORA_FILES["WOLNELEKTURY"],
    "ALL": CORPORA_FILES["ALL"],
    "PAN_TADEUSZ": CORPORA_FILES["PAN_TADEUSZ"],
}

file_name = "PAN_TADEUSZ"
files = FILES[file_name]


TOKENIZERS = {
    "all-corpora": "../tokenizer/tokenizers/all-corpora_32k.json",
    "wolne-lektury": "../tokenizer/tokenizers/wolne-lektury_32k.json",
    "pan-tadeusz": "../tokenizer/tokenizers/pan-tadeusz_32k.json",
    "deepseek-r1": "../tokenizer/tokenizers/deepseek-r1.json",
}
tokenizer_name = "pan-tadeusz"
tokenizer = Tokenizer.from_file(TOKENIZERS[tokenizer_name])

EMBEDDINGS_DIR = "./embeddings/"
model_path = f"{EMBEDDINGS_DIR}{file_name}_doc2vec_model_{tokenizer_name}.model"
model_d2v = Doc2Vec.load(model_path)


new_sentence = "Litwo! Ojczyzno moja! ty jesteś jak zdrowie."
print(f'Zdanie do wnioskowania: "{new_sentence}"')


sentence_map_path = (
    f"{EMBEDDINGS_DIR}{file_name}_doc2vec_sentence_map_{tokenizer_name}.json"
)

print("Wczytywanie tekstu z sentece cap...")
raw_sentences = []
with open(sentence_map_path, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
    raw_sentences.extend(lines)

if not raw_sentences:
    print("BŁĄD: Korpus danych jest pusty.")
    raise ValueError("Korpus danych jest pusty.")

loaded_model = model_d2v  # Używamy modelu prosto z treningu
sentence_lookup = raw_sentences  # Używamy listy zdań prosto z wczytywania korpusu


new_tokens = tokenizer.encode(new_sentence).tokens
print(f"\nTokeny dla zdania do wnioskowania: {new_tokens}")

inferred_vector = loaded_model.infer_vector(new_tokens, epochs=loaded_model.epochs)
print(f"\nWygenerowany wektor (embedding) dla zdania. Kształt: {inferred_vector.shape}")

# 3. Znajdowanie najbardziej podobnych wektorów z przestrzeni dokumentów/zdań
most_similar_docs = loaded_model.dv.most_similar([inferred_vector], topn=5)

print("\nnajbardziej podobne zdania z korpusu (Doc2Vec Inference):")
for doc_id_str, similarity in most_similar_docs:
    doc_index = int(doc_id_str)

    # Zabezpieczenie na wypadek błędu indeksowania (choć nie powinno wystąpić)
    try:
        original_sentence = sentence_lookup[doc_index]
        print(
            f"  - Sim: {similarity:.4f} | Zdanie (ID: {doc_id_str}): {
                original_sentence
            }"
        )
    except IndexError:
        print(
            f"  - Sim: {similarity:.4f} | BŁĄD: Nie znaleziono zdania dla ID: {doc_id_str}"
        )

tag_do_sprawdzenia = "4"  # Użyj formatu, jakiego używasz w mapie sentencji

# Wyszukiwanie podobnych dokumentów do WYTRENOWANEGO WEKTORA
similar_docs = loaded_model.dv.most_similar([tag_do_sprawdzenia], topn=5)

print(f"Wyniki podobieństwa dla dokumentu '{tag_do_sprawdzenia}':")
for tag, similarity in similar_docs:
    print(f"- Tag: {tag}, Similarity: {similarity:.10f}")
