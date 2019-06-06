import argparse
import json
import os
from typing import Dict, List

from mypy_extensions import TypedDict
from nltk.tokenize import word_tokenize
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Build a vocabulary out of COCO train2017 captions json file."
)

parser.add_argument(
    "-c",
    "--captions-jsonpath",
    default="data/coco/captions_train2017.json",
    help="Path to COCO train2017 captions json file.",
)
parser.add_argument("-t", "--word-count-threshold", type=int, default=5)
parser.add_argument(
    "-o",
    "--output-dirpath",
    default="data/vocabulary",
    help="Path to a (non-existent directory to save the vocabulary.",
)


# ------------------------------------------------------------------------------------------------
# All the punctuations in COCO captions, we will remove them.
# fmt: off
PUNCTUATIONS: List[str] = [
    "''", "#", "&", "$", "/", "'", "`", "(", ")", "{", "}", "?", "!", ":", "-", "...", ";", "."
]
# fmt: on

# Special tokens which should be added (all, or a subset) to the vocabulary.
# DO NOT write @@PADDING@@ token, AllenNLP would always add it internally.
SPECIAL_TOKENS: List[str] = ["@@UNKNOWN@@", "@start@", "@end@"]

# Type for each COCO caption example annotation.
CocoCaptionExample = TypedDict("CocoCaptionExample", {"id": int, "image_id": int, "caption": str})
# ------------------------------------------------------------------------------------------------


def build_caption_vocabulary(
    caption_json: List[CocoCaptionExample], word_count_threshold: int = 5
) -> List[str]:
    r"""
    Given a list of COCO caption examples, return a list of unique captions tokens thresholded
    by minimum occurence.
    """

    word_counts: Dict[str, int] = {}

    # Accumulate unique caption tokens from all caption sequences.
    for item in tqdm(caption_json):
        sequence: str = item["caption"].lower().strip()
        for punctuation in PUNCTUATIONS:
            sequence = sequence.replace(punctuation, "")

        sequence_tokens = word_tokenize(sequence)
        for token in sequence_tokens:
            if token in word_counts:
                word_counts[token] += 1
            else:
                word_counts[token] = 1

    caption_tokens = sorted(
        [key for key in word_counts if word_counts[key] >= word_count_threshold]
    )
    caption_vocabulary: List[str] = sorted(list(caption_tokens))
    return caption_vocabulary


if __name__ == "__main__":

    args = parser.parse_args()
    print(f"Loading annotations json from {args.captions_jsonpath}...")
    captions_json = json.load(open(args.captions_jsonpath))["annotations"]

    print("Building caption vocabulary...")
    caption_vocabulary: List[str] = build_caption_vocabulary(
        captions_json, args.word_count_threshold
    )
    caption_vocabulary = SPECIAL_TOKENS + caption_vocabulary
    print(f"Caption vocabulary size (with special tokens): {len(caption_vocabulary)}")

    # Write the vocabulary to separate namespace files in directory.
    print(f"Writing the vocabulary to {args.output_dirpath}...")
    print("Namespaces: tokens.")
    print("Non-padded namespaces: labels.")

    os.makedirs(args.output_dirpath, exist_ok=True)

    with open(os.path.join(args.output_dirpath, "tokens.txt"), "w") as f:
        for caption_token in caption_vocabulary:
            f.write(caption_token + "\n")

    with open(os.path.join(args.output_dirpath, "non_padded_namespaces.txt"), "w") as f:
        f.write("labels")
