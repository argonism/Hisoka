from transformers import AlbertTokenizer, BatchEncoding
from typing import List, Dict

def tokenize(texts: List[str],
             tokenizer: AlbertTokenizer,
             padding: bool = True,
             max_length: int = 512,
             truncation: bool = True,
             return_tensors: str = "pt",
             device: str = "cuda:0") -> BatchEncoding:

    tokenized = tokenizer(texts,
                            padding=padding,
                            truncation=truncation,
                            max_length=max_length,
                            return_tensors=return_tensors)
    tokenized.pop("token_type_ids")
    # tokenized.to(device)
    return tokenized

from pathlib import Path
import pickle
def to_pickle(obj, path):
    path = Path(path)
    path.write_bytes(pickle.dumps(obj))

def from_pickle(path):
    path = Path(path)
    return pickle.loads(path.read_bytes())

import json
def dump_dict_to_file(dic, path):
    path = Path(path)
    path.write_text(json.dumps(dic, indent=2, ensure_ascii=False))