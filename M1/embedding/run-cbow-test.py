import numpy as np
from tokenizers import Tokenizer
from gensim.models import Word2Vec

# --- DODANA FUNKCJA: OBLICZANIE WEKTORA DLA CAŁEGO SŁOWA ---


def get_word_vector_and_similar(
    word: str, tokenizer: Tokenizer, model: Word2Vec, topn: int = 20
):
    # Tokenizacja słowa na tokeny podwyrazowe
    # Używamy .encode(), aby otoczyć słowo spacjami, co imituje kontekst w zdaniu
    # Ważne: tokenizator BPE/SentencePiece musi widzieć spację, by dodać prefiks '_'
    encoding = tokenizer.encode(" " + word + " ")
    word_tokens = [t.strip() for t in encoding.tokens if t.strip()]  # Usuń puste tokeny

    # Usuwamy tokeny początku/końca sekwencji, jeśli zostały dodane przez tokenizator
    if word_tokens and word_tokens[0] in ["[CLS]", "<s>", "<s>", "Ġ"]:
        word_tokens = word_tokens[1:]
    if word_tokens and word_tokens[-1] in ["[SEP]", "</s>", "</s>"]:
        word_tokens = word_tokens[:-1]

    valid_vectors = []
    missing_tokens = []

    # 1. Zbieranie wektorów dla każdego tokenu
    for token in word_tokens:
        if token in model.wv:
            # Użycie tokenu ze spacją (np. '_ryż') lub bez (np. 'szlach')
            valid_vectors.append(model.wv[token])
        else:
            # W tym miejscu token może być zbyt rzadki i pominięty przez MIN_COUNT
            missing_tokens.append(token)

    if not valid_vectors:
        # Kod do obsługi, gdy żaden token nie ma wektora
        if missing_tokens:
            print(
                f"BŁĄD: Żaden z tokenów składowych ('{
                    word_tokens
                }') nie znajduje się w słowniku"
            )
        else:
            print(
                f"BŁĄD: Słowo '{
                    word
                }' nie zostało przetworzone na wektory (sprawdź tokenizację)."
            )
        return None, None

    # 2. Uśrednianie wektorów
    # Wektor dla całego słowa to średnia wektorów jego tokenów składowych
    word_vector = np.mean(valid_vectors, axis=0)

    # 3. Znalezienie najbardziej podobnych tokenów
    similar_words = model.wv.most_similar(positive=[word_vector], topn=topn)

    return word_vector, similar_words


TOKENIZERS = {
    "all-corpora": "../tokenizer/tokenizers/all-corpora_32k.json",
    "wolne-lektury": "../tokenizer/tokenizers/wolne-lektury_32k.json",
    "pan-tadeusz": "../tokenizer/tokenizers/pan-tadeusz_32k.json",
    "deepseek-r1": "../tokenizer/tokenizers/deepseek-r1.json",
}


tokenizer_name = "all-corpora"
corpora_name = "ALL"
tokenizer = Tokenizer.from_file(TOKENIZERS[tokenizer_name])

EMBEDDINGS_DIR = "./embeddings/cbow/"
model_path = f"{EMBEDDINGS_DIR}{corpora_name}_model_{tokenizer_name}_cbow.model"
model = Word2Vec.load(model_path)
# --- WERYFIKACJA UŻYCIA NOWEJ FUNKCJI ---


print(
    "\n--- Weryfikacja: Szukanie podobieństw dla całych SŁÓW (uśrednianie wektorów tokenów) ---"
)
print("Używany tokenizator: pan-tadeusz")
# Przykłady, które wcześniej mogły nie działać
words_to_test = [
    "wojsko",
    "szlachta",
]

for word in words_to_test:
    word_vector, similar_tokens = get_word_vector_and_similar(
        word, tokenizer, model, topn=10
    )

    if word_vector is not None:
        print(
            f"\n10 tokenów najbardziej podobnych do SŁOWA '{
                word
            }' (uśrednione wektory tokenów {tokenizer.encode(word).tokens}):"
        )
        # Wyświetlanie wektora (pierwsze 5 elementów)
        print(f"  > Wektor słowa (początek): {word_vector[:5]}...")
        for token, similarity in similar_tokens:
            print(f"  - {token}: {similarity:.4f}")

# --- WERYFIKACJA DLA WZORCA MATEMATYCZNEGO (Analogia wektorowa) ---

tokens_analogy = ["kobieta"]

# Używamy uśredniania wektorów dla tokenów
if tokens_analogy[0] in model.wv:
    similar_to_combined = model.wv.most_similar(positive=tokens_analogy, topn=10)

    print(f"\n10 tokenów najbardziej podobnych do kombinacji tokenów: {tokens_analogy}")
    for token, similarity in similar_to_combined:
        print(f"  - {token}: {similarity:.4f}")
else:
    print(
        f"\nOstrzeżenie: Co najmniej jeden z tokenów '{
            tokens_analogy
        }' nie znajduje się w słowniku. Pomięto analogię."
    )
