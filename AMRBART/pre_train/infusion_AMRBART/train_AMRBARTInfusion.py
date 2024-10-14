from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
)
from model_interface.tokenization_bart import AMRBartTokenizer

# from spring.spring_amr.tokenization_bart import PENMANBartTokenizer

# from AMRBART.pre_train.model_interface import BartForConditionalGeneration
from infusion_AMRBART.model import AMRBARTInfusion

from torch.optim.lr_scheduler import StepLR
from torch import nn
import numpy as np

import json
import torch
import argparse
import os


def read_inputfile(inputfile):
    data = []
    with open(inputfile, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            json_dict = json.loads(line)
            id = json_dict["id"]
            label = json_dict["label"]
            amr = json_dict["amr"]
            snt = json_dict["sent"]
            jp_snt = json_dict["jp_sent"]
            data.append([id, label, snt, jp_snt, amr])
    return data


def encode_amr(amr_tokenizer, amr_question, amr_article):
    input_amr = (
        [amr_tokenizer.bos_token]
        + amr_question.split()
        + [amr_tokenizer.eos_token]
        + amr_article.split()
        + [amr_tokenizer.eos_token]
    )

    token_ids = [
        amr_tokenizer.encoder.get(b, amr_tokenizer.unk_token_id) for b in input_amr
    ]
    # print("len of token_ids")
    # print((token_ids))
    # amr_input_ids = amr_tokenizer.tokenize_amr(input_amr)
    amr_attention_mask = [1] * len(token_ids)
    # amr_input = {"input_ids": amr_input_ids, "attention_mask": amr_attention_mask}
    amr_input = {"input_ids": token_ids, "attention_mask": amr_attention_mask}
    # input_ids, attention_mask
    return amr_input


def encode_text(text_tokenizer, text_question, text_article):
    text_input = text_tokenizer(
        text_question,
        text_article,
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=512,
    )
    # input_ids, attention_mask, token_type_ids
    return text_input


def prepare_input(
    question_path,
    article_path,
    text_tokenizer,
    amr_tokenizer,
    text_lang,
):
    # id, label, snt, jp_snt, amr
    question_data = read_inputfile(question_path)
    article_data = read_inputfile(article_path)
    data_len = len(question_data)

    data_samples = []

    for i in range(data_len):
        qid = question_data[i][0]
        label = question_data[i][1]

        # amr_input_ids, amr_attention_mask
        amr_input = encode_amr(amr_tokenizer, question_data[i][4], article_data[i][4])

        # text_input_ids, text_attention_mask, text_input_type_ids
        if text_lang == "en":
            text_input = encode_text(
                text_tokenizer, question_data[i][2], article_data[i][2]
            )
        elif text_lang == "jp":
            text_input = encode_text(
                text_tokenizer, question_data[i][3], article_data[i][3]
            )

        if label == "N":
            label = [[1], [0]]
        else:
            label = [[0], [1]]
        label = torch.tensor(label, dtype=torch.float32).view(-1, 2).cuda()
        data_samples.append([qid, label, text_input, amr_input])
    return data_samples


def accuracy(preds, labels):
    cnt = 0
    for i in range(len(preds)):
        if preds[i] == labels[i]:
            cnt += 1
    return cnt / len(preds)


def log_results(qids, probs, preds, labels, output_path):
    out = []
    for i in range(len(qids)):
        out.append(dict(id=qids[i], prob=probs[i], pred=preds[i], label=labels[i]))

    with open(output_path, "w") as f:
        f.write(json.dumps(out, indent=2))
    f.close()
    return 1


def train_one_epoch(model, train_ds, optimizer, scheduler):
    model.train()
    epoch_loss = 0

    for qid, label, text_input, amr_input in train_ds:
        optimizer.zero_grad()
        output = model(amr_input, text_input)
        loss = nn.CrossEntropyLoss()(output, label)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    scheduler.step()
    return epoch_loss / len(train_ds)


def eval(model, test_ds):
    eval_loss = 0
    qids = []
    probs = []
    preds = []
    labels = []

    model.eval()
    with torch.no_grad():
        for qid, label, text_input, amr_input in test_ds:
            output = model(amr_input, text_input)
            loss = nn.CrossEntropyLoss()(output, label)
            eval_loss += loss.item()

            probability = nn.Softmax(dim=1)(output)
            pred = torch.argmax(probability, dim=1).item()
            label = torch.argmax(label, dim=1).item()

            qids.append(qid)
            probs.append(probability.detach().cpu().numpy().tolist())
            preds.append(pred)
            labels.append(label)

        eval_loss = eval_loss / len(test_ds)
        eval_acc = accuracy(preds, labels)

    return eval_acc, eval_loss, qids, probs, preds, labels


def train_multiple_epoch(
    model, train_ds, test_ds, num_eps, learning_rate, log_path, save_path
):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer=optimizer, step_size=5, gamma=0.5)
    train_losses = []
    eval_losses = []
    eval_accs = []
    eval_logs = []
    for epoch in range(num_eps):
        checkpoint_path = save_path
        train_losses.append(train_one_epoch(model, train_ds, optimizer, scheduler))
        eval_acc, eval_loss, qids, probs, preds, labels = eval(model, test_ds)

        print("eval loss: ", eval_loss)
        print("eval acc: ", eval_acc)

        eval_losses.append(eval_loss)
        eval_accs.append(eval_acc)
        eval_logs.append([qids, probs, preds, labels])

        checkpoint_path += f"/epoch_{epoch}.pt"
        torch.save(model.state_dict(), checkpoint_path)

    best_epoch = np.argmax(eval_accs)
    print("best epoch: ", best_epoch)
    output_path = os.path.join(log_path, "best_epoch")
    log_results(
        qids=eval_logs[best_epoch][0],
        probs=eval_logs[best_epoch][1],
        preds=eval_logs[best_epoch][2],
        labels=eval_logs[best_epoch][3],
        output_path=output_path,
    )
    return train_losses, eval_losses, eval_accs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model_type",
        required=True,
        type=str,
        choices=["only_amr", "infusion"],
    )
    parser.add_argument(
        "-infusion_type", required=False, type=str, choices=["concat", "co_attn_res"]
    )
    parser.add_argument("-text_lang", required=True, type=str, choices=["en", "jp"])
    parser.add_argument("-parser", required=True, type=str, choices=["gsp", "spring"])
    parser.add_argument("-data_year", required=True, type=int, choices=[2023, 2024])
    parser.add_argument(
        "-log_path",
        required=False,
        type=str,
        default="/home/yenvth/workspace/minhnt-thesis/AMRBART/pre_train/infusion_AMRBART/ckpt/checkpoint/",
    )
    parser.add_argument(
        "-save_path",
        required=False,
        type=str,
        default="/home/yenvth/workspace/minhnt-thesis/AMRBART/pre_train/infusion_AMRBART/ckpt/checkpoint/",
    )
    parser.add_argument("-lr", required=False, type=float, default=0.001)
    parser.add_argument("-num_eps", required=False, type=int, default=10)
    parser.add_argument("-description", required=False, type=str)
    args = parser.parse_args()

    output_folder = os.path.join(
        args.log_path,
        f"ambart_{args.parser}_{args.model_type}_{args.infusion_type}_{args.text_lang}_{args.data_year}",
    )
    print(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(args.description)

    amr_pretrained_path = "xfbai/AMRBART-base-finetuned-AMR2.0-AMRParsing"
    # amr_pretrained_path = "xfbai/AMRBART-base"
    amr_model = AutoModel.from_pretrained(amr_pretrained_path)
    amr_tokenizer = AMRBartTokenizer.from_pretrained(amr_pretrained_path)
    amr_config = AutoConfig.from_pretrained(amr_pretrained_path)

    text_pretrained_path = "google-bert/bert-base-multilingual-uncased"
    text_model = AutoModel.from_pretrained(text_pretrained_path)
    text_tokenizer = AutoTokenizer.from_pretrained(text_pretrained_path)

    concat_emb_dim = 768 * 2
    emb_dim = 768
    model = AMRBARTInfusion(
        text_model=text_model,
        amr_model=amr_model,
        amr_eos_token_id=amr_config.eos_token_id,
        dropout=0.1,
        mode=args.model_type,
        concat_emb_dim=concat_emb_dim,
        emb_dim=emb_dim,
        infusion_type=args.infusion_type,
    )

    model.to("cuda")
    summary_model(model)

    if args.data_year == 2024:
        if args.parser == "spring":
            train_question_path = (
                "infusion_AMRBART/data/spring/train2024_question_v2.jsonl"
            )
            train_article_path = (
                "infusion_AMRBART/data/spring/train2024_article_v2.jsonl"
            )
            test_question_path = (
                "infusion_AMRBART/data/spring/test2024_question_v2.jsonl"
            )
            test_article_path = "infusion_AMRBART/data/spring/test2024_article_v2.jsonl"
        elif args.parser == "gsp":
            train_question_path = (
                "infusion_AMRBART/data/gsp_parser/train2024_question_v2.jsonl"
            )
            train_article_path = (
                "infusion_AMRBART/data/gsp_parser/train2024_article_v2.jsonl"
            )
            test_question_path = (
                "infusion_AMRBART/data/gsp_parser/test2024_question_v2.jsonl"
            )
            test_article_path = (
                "infusion_AMRBART/data/gsp_parser/test2024_article_v2.jsonl"
            )

    elif args.data_year == 2023:
        if args.parser == "spring":
            train_question_path = (
                "infusion_AMRBART/data/spring/train2023_question_v2.jsonl"
            )
            train_article_path = (
                "infusion_AMRBART/data/spring/train2023_article_v2.jsonl"
            )
            test_question_path = (
                "infusion_AMRBART/data/spring/test2023_question_v2.jsonl"
            )
            test_article_path = "infusion_AMRBART/data/spring/test2023_article_v2.jsonl"
        elif args.parser == "gsp":
            train_question_path = (
                "infusion_AMRBART/data/gsp_parser/train2023_question_v2.jsonl"
            )
            train_article_path = (
                "infusion_AMRBART/data/gsp_parser/train2023_article_v2.jsonl"
            )
            test_question_path = (
                "infusion_AMRBART/data/gsp_parser/test2023_question_v2.jsonl"
            )
            test_article_path = (
                "infusion_AMRBART/data/gsp_parser/test2023_article_v2.jsonl"
            )

    # qid, label, text_input, amr_input
    train_ds = prepare_input(
        question_path=train_question_path,
        article_path=train_article_path,
        text_tokenizer=text_tokenizer,
        amr_tokenizer=amr_tokenizer,
        text_lang=args.text_lang,
    )

    test_ds = prepare_input(
        question_path=test_question_path,
        article_path=test_article_path,
        text_tokenizer=text_tokenizer,
        amr_tokenizer=amr_tokenizer,
        text_lang=args.text_lang,
    )

    print(len(train_ds), len(test_ds))

    train_losses, test_losses, test_accs = train_multiple_epoch(
        model=model,
        train_ds=train_ds,
        test_ds=test_ds,
        num_eps=args.num_eps,
        learning_rate=args.lr,
        log_path=output_folder,
        save_path=output_folder,
    )
    print("train_losses: ", train_losses)
    print("eval_losses: ", test_losses)
    print("test accuracy: ", test_accs)


def summary_model(model):
    num_para = 0
    for p in model.parameters():
        if p.requires_grad:
            num_para += p.numel()
    print(num_para)
    # print(model)
    return 1


def test():
    amr_pretrained_path = "xfbai/AMRBART-base-finetuned-AMR2.0-AMRParsing"
    # amr_pretrained_path = "xfbai/AMRBART-base"
    amr_model = AutoModel.from_pretrained(amr_pretrained_path)
    amr_tokenizer = AMRBartTokenizer.from_pretrained(amr_pretrained_path)
    t = encode_amr(amr_tokenizer, "test question", "test article")
    print(t)


if __name__ == "__main__":
    main()
    # test()
