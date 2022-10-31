import torch
from model import BertDenseEncoder
from transformers import BertJapaneseTokenizer, BertModel

def main(query, document):
    # query = [
    #     "情報検索とは何ですか",
    #     "情報検索とは何ですか",
    # ]
    # document = [
    #     "Microsoft の機械読解 (MRC) がテキストから意味をスキャンして抽出する方法について説明します。",
    #     "情報検索（じょうほうけんさく、英語: Information retrieval）とは、コンピュータを用いて大量のデータ群から目的に合致したものを取り 出すこと。",
    # ]
    if isinstance(query, str): query = [query]
    if isinstance(document, str): document = [document]

    path = "./model/checkpoint-15000"
    tokenizer = BertJapaneseTokenizer.from_pretrained(path)
    tokenizer.add_tokens("[Q]", special_tokens=True)
    tokenizer.add_tokens("[D]", special_tokens=True)

    encoder = BertDenseEncoder.from_pretrained(path, tokenizer=tokenizer)

    with torch.no_grad():
        max_length = 512
        query_tokenized = tokenizer(["[Q]"+text for text in query],
                               padding=True,
                               truncation=True,
                               max_length=max_length,
                               return_tensors='pt')
        document_tokenized = tokenizer(["[D]"+text for text in document],
                               padding=True,
                               truncation=True,
                               max_length=max_length,
                               return_tensors='pt')
        # query_emb = encoder.get_emb(query_tokenized)
        # doc_emb = encoder.get_emb(document_tokenized)

        output = encoder(query_tokenized, positive=document_tokenized)

        # query_outputs = encoder.encoder(**query_tokenized)
        # query_emb = query_outputs.last_hidden_state[:, 0]
        # query_emb = encoder.linear(query_emb)

        # positive_outputs = encoder.encoder(**document_tokenized)
        # pos_emb = positive_outputs.last_hidden_state[:, 0]
        # pos_emb = encoder.linear(pos_emb)
        # print(pos_emb[0], pos_emb.shape)
    print(output.logits)

def handle_args():
    import argparse

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('query')
    parser.add_argument('document')
    # parser.add_argument('--arg3')
    # parser.add_argument('-a', '--arg4')

    return parser.parse_args()

if __name__ == "__main__":
    args = handle_args()
    main(args.query, args.document)
    