import argparse
import torch
import torch_mlu
import json
import evaluate
from tqdm import tqdm

import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

def get_parser():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--file_in", "-i", type=str, default="")
    parser.add_argument("--file_out", "-o", type=str, default="")
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    args = parser.parse_args()
    return args

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # model_output的第一个元素包含所有的token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def process_batch(batch, model, tokenizer, model_mpnet, tokenizer_mpnet, metric, device, fo):
    # 翻译任务
    ori_texts = [line['ori_text'] for line in batch]
    inputs = tokenizer(ori_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = inputs.to(device)
    # print(inputs["input_ids"])
    outputs = model.generate(inputs["input_ids"], max_length=256, num_beams=4, early_stopping=True)
    mt_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # 构造句子向量输入
    ori_mpnet = []
    sentences = [line['sentence'] for line in batch]
    for sentence, mt_text in zip(sentences, mt_texts):
        ori_mpnet.append(sentence)
        ori_mpnet.append(mt_text)

    # encoded_input 奇数行为原句，偶数行为目标句
    inputs_mpnet= tokenizer_mpnet(ori_mpnet, padding=True, truncation=True, return_tensors='pt')
    inputs_mpnet = inputs_mpnet.to(device)
    
    # 计算token embeddings
    with torch.no_grad():
        outputs_mpnet = model_mpnet(**inputs_mpnet)
        
    # 进行池化操作
    sentence_embeddings = mean_pooling(outputs_mpnet, inputs_mpnet['attention_mask'])
    
    # 标准化embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    cos_sims = F.cosine_similarity(sentence_embeddings[::2], sentence_embeddings[1::2], dim=1)
    
    for line, cos_sim, mt_text in zip(batch, cos_sims, mt_texts):
        line['cos_sim'] = f"{cos_sim:.3f}"
        line['mt_text'] = mt_text
        if False:
            # 计算 ter 指标
            score = metric.compute(
                predictions=[mt_text], references=[line['sentence']], case_sensitive=True, ignore_punct=True)['score']
            line['ter'] = f"{score:.2f}"
        print(json.dumps(line, ensure_ascii=False), file=fo, flush=True)


def main():
    device="mlu"
    metric = evaluate.load("ter")

    mtModel="./opus-mt-zh-en"
    tokenizer=AutoTokenizer.from_pretrained(mtModel)
    model=AutoModelForSeq2SeqLM.from_pretrained(mtModel)
    model=model.to(device)

    Model_mpnet = "./all-mpnet-base-v2"
    tokenizer_mpnet = AutoTokenizer.from_pretrained(Model_mpnet)
    model_mpnet = AutoModel.from_pretrained(Model_mpnet)
    model_mpnet=model_mpnet.to(device)

    args = get_parser()
    batch_size = args.batch_size
    with open(args.file_in, 'r') as fi, open(args.file_out, 'w') as fo:
        total_lines = sum(1 for _ in fi)
        fi.seek(0)
        batch = []
        for line in tqdm(fi, desc=f"running", total=total_lines, colour='green'):
            line = json.loads(line.strip())
            batch.append(line)
            if len(batch) == batch_size:
                process_batch(batch, model, tokenizer, model_mpnet, tokenizer_mpnet, metric, device, fo)
                batch = []
        if batch:  # process the last batch if it is not empty
            process_batch(batch, model, tokenizer, model_mpnet, tokenizer_mpnet, metric, device, fo)



if __name__ == '__main__':
    main()
