from typing import Union
from pydantic import BaseModel
from datasets import load_dataset
import torch
from transformers import TrainingArguments
from transformers import Trainer
from transformers import EarlyStoppingCallback

from tqdm import tqdm
from pathlib import Path
import logging
import json
from datetime import datetime

from model import setup_model_tokenizer
from utils import tokenize, dump_dict_to_file, load_json_from


class MMarcoCollator():
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _tokenize(self, texts):
        tokenized = tokenize(texts, self.tokenizer)
        return tokenized

    def _extract(self, key, examples, prefix):
        return list(map(lambda x: prefix+x[key], examples))
    
    def __call__(self, examples):
        encodings = {
            'query': self._tokenize(self._extract("query", examples, "[Q]")),
            'positive': self._tokenize(self._extract("positive", examples, "[D]")),
            'negative': self._tokenize(self._extract("negative", examples, "[D]")),
        }

        return encodings

class ModelArgs(BaseModel):
    model_name_or_path: str
    batch_size: int

class MiscArgs(BaseModel):
    seed: int = 42
    logging_steps: int = 50

class TrainArgs(ModelArgs, MiscArgs):
    model_name_or_path: str
    train_size: int
    eval_size: int
    num_epoch: int
    dataset_offset: int
    resume_from_checkpoint: Union[bool, None]
    output_dir: str

class EvalArgs(ModelArgs, MiscArgs):
    eval_size: int

def train(args: TrainArgs):

    model_name_or_path = args.model_name_or_path
    encoder, tokenizer = setup_model_tokenizer(model_name_or_path)

    mmarco_collator = MMarcoCollator(tokenizer)

    dataset = load_dataset('unicamp-dl/mmarco', 'japanese')["train"]

    seed = args.seed
    dataset_offset = args.dataset_offset
    train_size = dataset_offset + args.train_size
    eval_size = args.eval_size

    train_dataset = dataset.shuffle(seed=seed).select(range(train_size))
    eval_dataset = dataset.shuffle(seed=seed).select(range(train_size, train_size+eval_size))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy='epoch',
        logging_strategy='steps',
        logging_steps=50,
        save_strategy='epoch',
        save_total_limit=1,
        lr_scheduler_type='constant',
        load_best_model_at_end=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epoch,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=encoder,
        tokenizer=tokenizer,
        data_collator=mmarco_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train(ignore_keys_for_eval=['last_hidden_state', 'hidden_states', 'attentions'],
                  resume_from_checkpoint=args.resume_from_checkpoint)

def eval_beir(args: EvalArgs):
    from beir import LoggingHandler
    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval
    from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
    from beir.retrieval.search.lexical import BM25Search as BM25

    logging.basicConfig(format='%(asctime)s - %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[LoggingHandler()])

    dataset = "mrtydi/japanese"
    script_dir = Path(__file__).parent.absolute()
    data_path = Path(script_dir.parent.absolute(), f"datasets/{dataset}")

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    k_values = [1, 10, 100]

    hostname = "localhost"
    index_name = dataset.replace("/", "-")
    initialize = True
    bm25 = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
    retriever = EvaluateRetrieval(bm25, k_values=k_values)

    bm25_cache_path = Path(f"bm25_{dataset}_{max(k_values)}.json")
    bm25_results = {}
    if bm25_cache_path.exists():
        bm25_results = load_json_from(bm25_cache_path)
    else:
        bm25_results = retriever.retrieve(corpus, queries)
        dump_dict_to_file(bm25_results, bm25_cache_path)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, bm25_results, k_values=k_values)
 
    encoder, tokenizer = setup_model_tokenizer(args.model_name_or_path, mode="eval", device="cuda:0")
    model = DRES(encoder, batch_size=args.batch_size)

    dense_retriever = EvaluateRetrieval(model, score_function="dot")
    rerank_results = dense_retriever.rerank(corpus, queries, bm25_results, top_k=100)

    ndcg, _map, recall, precision = dense_retriever.evaluate(qrels, rerank_results, k_values=k_values)
    eval_result = {
        "ndcg": ndcg, "map": _map, "recall": recall, "precision": precision
    }

    eval_result_path = script_dir.joinpath("eval_result", f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    eval_result_path.write_text(json.dumps(eval_result, indent=2, ensure_ascii=False))

from argparse import ArgumentParser
def parse_known_args():
    parser = ArgumentParser(description='')
    parser.add_argument('mode', help='{train or eval}')
    parser.add_argument('-m', '--model_name_or_path')
    parser.add_argument('-t', '--train_size', default=30000)
    parser.add_argument('-b', '--batch_size', default=2)
    parser.add_argument('-e', '--eval_size', default=3000)
    parser.add_argument('--dataset_offset', default=0)
    parser.add_argument('-n', '--num_epoch', default=1)
    parser.add_argument('-o', '--output_dir', default="train_output")
    parser.add_argument('--resume_from_checkpoint', default=None)
    

    return parser.parse_known_args()

if __name__ == "__main__":
    args, other = parse_known_args()
    args_dict = vars(args)
    if args.mode == "train":
        train_args = TrainArgs(**args_dict)
        train(train_args)
    elif args.mode == "eval":
        eval_args = EvalArgs(**args_dict)
        eval_beir(eval_args)
    else:
        raise ValueError("You need to specify 'train' or 'eval'")