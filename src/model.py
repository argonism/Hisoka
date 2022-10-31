import torch
from torch import nn
from torch.nn import TripletMarginWithDistanceLoss
from transformers import BertPreTrainedModel, DistilBertModel, AlbertTokenizer
from transformers.modeling_outputs import ModelOutput

def setup_model_tokenizer(model_name_or_path: str) -> tuple[BertPreTrainedModel, AlbertTokenizer]:
    tokenizer = AlbertTokenizer.from_pretrained(model_name_or_path)
    tokenizer.add_tokens("[Q]", special_tokens=True)
    tokenizer.add_tokens("[D]", special_tokens=True)

    encoder = BertDenseEncoder.from_pretrained(model_name_or_path, tokenizer=tokenizer)
    return encoder, tokenizer

class BertDenseEncoder(BertPreTrainedModel):
    def __init__(self, config, tokenizer=None):
        super().__init__(config)

        self.embedding_dim = 128
        self.encoder = DistilBertModel(config)
        # if tokenizer is not None:
        #     self.encoder.resize_token_embeddings(len(tokenizer))
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.linear = nn.Linear(config.hidden_size, self.embedding_dim)

        self.post_init()
        # self.similarity_func = nn.CosineSimilarity()
        # self.loss_function = TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1 - self.similarity_func(x, y))

    @classmethod
    def calc_similarity(cls, vectors1, vectors2):
        return torch.bmm(vectors1.unsqueeze(1), torch.transpose(vectors2.unsqueeze(1), 1, 2))[:, 0]

    def get_emb(self, tokenized):
        outputs = self.encoder(**tokenized)
        embedding = outputs.last_hidden_state[:, 0]
        return embedding

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
