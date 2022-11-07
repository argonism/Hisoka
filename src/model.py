import abc
from typing import List, Dict

import torch
from torch import nn
from transformers import BertPreTrainedModel, DistilBertModel, AlbertTokenizer
from transformers.modeling_outputs import ModelOutput

import numpy as np
from utils import tokenize

def setup_model_tokenizer(model_name_or_path: str, mode="train", device="cuda:0") -> tuple[BertPreTrainedModel, AlbertTokenizer]:
    tokenizer = AlbertTokenizer.from_pretrained(model_name_or_path)
    if mode == "train":
        tokenizer.add_tokens("[Q]", special_tokens=True)
        tokenizer.add_tokens("[D]", special_tokens=True)

    encoder = BertDenseEncoder.from_pretrained(model_name_or_path, tokenizer=tokenizer)
    # encoder.to(device)
    return encoder, tokenizer

class IBEIRDenseEncoder(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        pass

    @abc.abstractmethod
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        pass

class BertDenseEncoder(BertPreTrainedModel, IBEIRDenseEncoder):
    def __init__(self, config, tokenizer=None):
        super().__init__(config)

        self.embedding_dim = 256
        self.encoder = DistilBertModel(config)
        self.tokenizer = tokenizer
        if tokenizer is not None and not config.vocab_size == len(self.tokenizer):
            self.encoder.resize_token_embeddings(len(self.tokenizer))
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.linear = nn.Linear(config.hidden_size, self.embedding_dim)

        self.post_init()

    @classmethod
    def calc_similarity(cls, vectors1, vectors2):
        # return torch.mm(vectors1, vectors2.transpose(0, 1))
        return torch.bmm(vectors1.unsqueeze(1), torch.transpose(vectors2.unsqueeze(1), 1, 2))[:, 0]

    def encode(self, texts, batch_size) -> np.ndarray:
        from tqdm import tqdm
        texts_len = len(texts)
        embs = []
        for batch_idx in tqdm(range(0, texts_len, batch_size)):
            batch = texts[batch_idx:min(batch_idx+batch_size, texts_len)]
            tokenized = tokenize(batch, self.tokenizer, device=self.device)

            outputs = self.encoder(**tokenized)
            emb = outputs.last_hidden_state[:, 0]
            emb = self.linear(emb)
            emb = emb.detach().cpu()
            embs.append(emb)

        return np.concatenate(embs)

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        queries = ["[Q]" + query for query in queries]
        query_embs = self.encode(queries, batch_size)
        return query_embs

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        sentences = [("[D]"+ doc["title"] + " " + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        corpus_emb = self.encode(sentences, batch_size)
        return corpus_emb

    def calc_loss(self, query, positive, negative): 
        sim_func = self.__class__.calc_similarity
    
        pos_sim = sim_func(query, positive)
        neg_sim = sim_func(query, negative)
        logit_matrix = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(logit_matrix.size(0), dtype=torch.long, device="cuda:0")
        return nn.CrossEntropyLoss()(logit_matrix, labels)

    def forward(self,
                query,
                positive=None,
                negative=None,
                attention_mask=None,
                position_ids=None,
                token_type_ids=None,
                output_attentions=False,
                output_hidden_states=False):
        
        query_outputs = self.encoder(**query)
        query_emb = query_outputs.last_hidden_state[:, 0]
        query_emb = self.linear(query_emb)

        assert positive is not None
        positive_outputs = self.encoder(**positive)
        pos_emb = positive_outputs.last_hidden_state[:, 0]
        pos_emb = self.linear(pos_emb)

        score = self.calc_similarity(query_emb, pos_emb)
        neg_emb = None
        if negative is not None:
            negative_outputs = self.encoder(**negative)
            neg_emb = negative_outputs.last_hidden_state[:, 0]
            neg_emb = self.linear(neg_emb)

        loss = None
        if neg_emb is not None:
            loss = self.calc_loss(query_emb, pos_emb, neg_emb)

        attentions=None
        if output_attentions:
            attentions=query_outputs.attentions
        
        hidden_states=None
        if output_hidden_states:
            hidden_states=query_outputs.hidden_states
        
        return ModelOutput(
            logits=score,
            loss=loss,
            last_hidden_state=query_outputs.last_hidden_state,
            attentions=attentions,
            hidden_states=hidden_states
        )
