import argparse
import torch
import torch_mlu
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


def get_parser():
    parser = argparse.ArgumentParser(description='本脚本的说明')
    parser.add_argument("--file_in", "-i", type=str, default="")
    parser.add_argument("--file_out", "-o", type=str, default="")
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    args = parser.parse_args()
    return args


# 平均池化 - 考虑注意力掩码进行正确的平均
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # model_output的第一个元素包含所有的token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def process_batch(batch, ori_texts, model, tokenizer, device, fo):
    # encoded_input 奇数行为原句，偶数行为目标句
    encoded_input = tokenizer(ori_texts, padding=True, truncation=True, return_tensors='pt')
    encoded_input = encoded_input.to(device)
    
    # 计算token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        
    # 进行池化操作
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    
    # 标准化embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    cos_sims = F.cosine_similarity(sentence_embeddings[::2], sentence_embeddings[1::2], dim=1)
    
    for line, cos_sim in zip(batch, cos_sims):
        line['cos_sim'] = f"{cos_sim:.3f}"
        print(json.dumps(line, ensure_ascii=False), file=fo, flush=True)


def main():
    mtModel = "./all-mpnet-base-v2"
    device = "mlu"
    tokenizer = AutoTokenizer.from_pretrained(mtModel)
    model = AutoModel.from_pretrained(mtModel)
    model = model.to(device)
    model.eval()

    args = get_parser()
    batch_size = args.batch_size
    with open(args.file_in, 'r') as fi, open(args.file_out, 'w') as fo:
        total_lines = sum(1 for _ in fi)
        fi.seek(0)
        batch = []
        ori_texts = []
        for line in tqdm(fi, total=total_lines):
            line = json.loads(line.strip())
            batch.append(line)
            ori_texts.append(line['sentence'])
            ori_texts.append(line['mt_text'])
            if len(batch) == batch_size:
                process_batch(batch, ori_texts, model, tokenizer, device, fo)
                batch = []
                ori_texts = []
        if batch:  # 如果最后一个batch不为空，处理这个batch
            process_batch(batch, ori_texts, model, tokenizer, device, fo)


if __name__ == '__main__':
    main()
