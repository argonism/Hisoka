from transformers import AlbertTokenizer, BatchEncoding
from typing import List, Dict

def tokenize(texts: List[str],
             tokenizer: AlbertTokenizer,
             padding: bool = True,
             max_length: bool = True,
             truncation: bool = True,
             return_tensors: str = "pt") -> BatchEncoding:

    tokenized = tokenizer(texts,
                            padding=padding,
                            truncation=truncation,
                            max_length=max_length,
                            return_tensors=return_tensors)
    tokenized.pop("token_type_ids")
    return tokenized

