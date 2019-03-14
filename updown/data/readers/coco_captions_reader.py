import json
from typing import Any, Dict, List

from nltk.tokenize import word_tokenize


class CocoCaptionsReader(object):
    def __init__(self, captions_jsonpath: str):
        self._captions_jsonpath = captions_jsonpath

        captions_json: Dict[str, Any] = json.load(open(self._captions_jsonpath))
        PUNCTUATIONS = ["'", "`", "(", ")", "{", "}", ".", "?", "!", ",", ":", "-", "...", ";"]

        # Build an index with keys as image IDs and values as list of tokenized captions.
        self._image_id_to_caption_tokens: Dict[int, List[List[str]]] = {}

        for caption_item in captions_json["annotations"]:
            if caption_item["image_id"] not in self._image_id_to_caption_tokens:
                self._image_id_to_caption_tokens[caption_item["image_id"]] = []

            caption: str = caption_item["caption"].lower()
            for punctuation in PUNCTUATIONS:
                caption = caption.replace(punctuation, "")

            caption_tokens: List[str] = word_tokenize(caption)
            self._image_id_to_caption_tokens[caption_item["image_id"]].append(caption_tokens)

    def __len__(self):
        return len(self._image_id_to_caption_tokens)

    def __getitem__(self, image_id):
        return self._image_id_to_caption_tokens[image_id]
