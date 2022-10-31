from typing import Union
from pydantic import BaseModel
from datasets import load_dataset
import torch
from transformers import TrainingArguments
from transformers import Trainer
from transformers import EarlyStoppingCallback

from model import BertDenseEncoder, setup_model_tokenizer

from transformers import BertJapaneseTokenizer
class MMarcoCollator():
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _tokenize(self, texts):
        tokenized = self.tokenizer(texts,
                               padding=True,
                               truncation=True,
                               max_length=self.max_length,
                               return_tensors='pt')
        tokenized.pop("token_type_ids")
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

class EvalArgs(ModelArgs, MiscArgs):
    eval_size: int

def train(args: TrainArgs):

    # model_name_or_path = "cl-tohoku/bert-base-japanese-v2"
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
        output_dir='./model/',
        evaluation_strategy='steps',
        eval_steps=5000, 
        logging_strategy='steps',
        logging_steps=50,
        save_strategy='steps',
        save_steps=5000,
        save_total_limit=1,
        lr_scheduler_type='constant',
        load_best_model_at_end=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        remove_unused_columns=False
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

from tqdm import tqdm
def eval(args: EvalArgs):

    model_name_or_path = args.model_name_or_path
    encoder, tokenizer = setup_model_tokenizer(model_name_or_path)
    encoder.eval()
    encoder.to("cuda:0")

    mmarco_collator = MMarcoCollator(tokenizer)

    dataset = load_dataset('unicamp-dl/mmarco', 'japanese')["train"]

    seed = args.seed
    eval_size = args.eval_size
    batch_size = args.batch_size

    dataset = dataset.shuffle(seed=seed).select(range(eval_size))
    dataset_iter = iter(dataset)

    success, sum_ = 0, 0
    with torch.no_grad():
        for step in tqdm(range(len(dataset) // batch_size)):
            tokenized = mmarco_collator([next(dataset_iter) for _ in range(batch_size)])
            tokenized = {key:value.to("cuda:0") for key, value in tokenized.items()}
            # out = encoder(**tokenized)
            # print(out.loss)
            query = tokenized["query"]
            positive = tokenized["positive"]
            negative = tokenized["negative"]
            pos_sim = encoder(query, positive=positive).logits
            neg_sim = encoder(query, positive=negative).logits
            for example in pos_sim > neg_sim:
                if example[0]: success += 1
                sum_ += 1
    print(success, sum_, success / sum_)

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
        eval(eval_args)
    else:
        raise ValueError("You need to specify 'train' or 'eval'")