from tabulate import tabulate
from tokenizers import Tokenizer
from corpora import get_corpus_file
from pathlib import Path

TOKENIZERS = {
    # "bpe": "tokenizers/custom_bpe_tokenizer.json",
    "bielik-v1": "tokenizers/bielik-v1-tokenizer.json",
    "bielik-v2": "tokenizers/bielik-v2-tokenizer.json",
    "bielik-v3": "tokenizers/bielik-v3-tokenizer.json",
    "pan-tadeusz_16k": "tokenizers/pan-tadeusz_16k.json",
    "pan-tadeusz_32k": "tokenizers/pan-tadeusz_32k.json",
    "wolne-lektury_16k": "tokenizers/wolne-lektury_16k.json",
    "wolne-lektury_32k": "tokenizers/wolne-lektury_32k.json",
    "nkjp_16": "tokenizers/nkjp_16k.json",
    "nkjp_32": "tokenizers/nkjp_32k.json",
    "all-corpora_16": "tokenizers/all-corpora_16k.json",
    "all-corpora_32": "tokenizers/all-corpora_32k.json",
    "deepseek-r1": "tokenizers/deepseek-r1.json",
}


FILES = {
    "Pan Tadeusz": get_corpus_file("WOLNELEKTURY", "pan-tadeusz-ksiega-1.txt")[0],
    "Pickwick Papers": Path("../korpus-mini/the-pickwick-papers-gutenberg.txt"),
    "Fryderyk Chopin (wiki)": Path("../korpus-mini/fryderyk-chopin-wikipedia.txt"),
}


# def tokenize_files(tokenizer: Tokenizer):
#     for file_path in FILES:
#         print(f"Tokenizing file: {file_path}")
#         source_txt = ""
#         with open(file_path, "r", encoding="utf-8") as f:
#             source_txt = f.read()
#
#         encoded = tokenizer.encode(source_txt)
#
#         return encoded
#
def tokenize_file(tokenizer: Tokenizer, file_name: str, file_path: Path):
    source_txt = ""
    with open(file_path, "r", encoding="utf-8") as f:
        source_txt = f.read()

    encoded = tokenizer.encode(source_txt)

    return {"tokens_count": len(encoded.tokens), file_name: file_name}


def visualize_tokenization_results():
    results = {file_name: [] for file_name in FILES.keys()}

    for tokenizer_name, tokenizer_path in TOKENIZERS.items():
        print(f"Loading tokenizer: {tokenizer_name} from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path)

        for file_name, file_path in FILES.items():
            tokenization_result = tokenize_file(tokenizer, file_name, file_path)
            tokenization_result["Tokenizer"] = tokenizer_name
            results[file_name].append(tokenization_result)

    for file_name, tokenization_results in results.items():
        print(f"\nTokenization results for file: {file_name}")
        # Sort by tokens_count
        tokenization_results = sorted(
            tokenization_results, key=lambda x: x["tokens_count"]
        )
        # Create progress bar
        max_tokens = max([res["tokens_count"] for res in tokenization_results])
        table_data = []
        for res in tokenization_results:
            progress_bar_length = int((res["tokens_count"] / max_tokens) * 20)
            progress_bar = (
                "[" + "#" * progress_bar_length + " " * (20 - progress_bar_length) + "]"
            )
            table_data.append([res["Tokenizer"], progress_bar, res["tokens_count"]])

        print(
            tabulate(
                table_data,
                headers=["Tokenizer", "Progress", "Tokens Count"],
                tablefmt="github",
            )
        )
    return results


if __name__ == "__main__":
    visualize_tokenization_results()
