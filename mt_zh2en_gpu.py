import argparse
import torch
import torch_mlu
import json
from tqdm import tqdm
import evaluate
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader

class JsonDataset(Dataset):
    def __init__(self, filepath, tokenizer):
        self.filepath = filepath
        self.tokenizer = tokenizer
        with open(filepath, 'r') as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = json.loads(self.lines[idx].strip())
        inputs = self.tokenizer(line['ori_text'], return_tensors="pt", truncation=True, max_length=512)
        return line, inputs


def get_parser():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--file_in", "-i", type=str, default="")
    parser.add_argument("--file_out", "-o", type=str, default="")
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--num_workers", "-n", type=int, default=0)
    args = parser.parse_args()
    return args

def collate_fn(batch):
    keys = batch[0][1].keys()
    padded_batch = {}
    for key in keys:
        tensors = [item[1][key].squeeze(0) for item in batch]
        padded_tensors = pad_sequence(tensors, batch_first=True)
        padded_batch[key] = padded_tensors

    padded_batch['lines'] = [item[0] for item in batch]
    return padded_batch

def process_batch(batch, model, tokenizer, metric, device, fo):
    inputs = batch["input_ids"].to(device)
    lines = batch["lines"]
    outputs = model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
    mt_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    for line, mt_text in zip(lines, mt_texts):
        score = metric.compute(
            predictions=[mt_text], references=[line['sentence']], case_sensitive=True, ignore_punct=True)['score']
        line['mt_text'] = mt_text
        line['ter'] = f"{score:.2f}"
        print(json.dumps(line, ensure_ascii=False), file=fo, flush=True)


def main():
    metric = evaluate.load("ter")
    mtModel="./opus-mt-zh-en"
    device="mlu"
    tokenizer=AutoTokenizer.from_pretrained(mtModel)
    model=AutoModelForSeq2SeqLM.from_pretrained(mtModel)
    model=model.to(device)
    model.eval()
    args = get_parser()
    dataset = JsonDataset(args.file_in, tokenizer)
    # 在DataLoader中使用collate_fn
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers)

    with open(args.file_out, 'w') as fo:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='running', colour='green'):
                process_batch(batch, model, tokenizer, metric, device, fo)



if __name__ == '__main__':
    main()
