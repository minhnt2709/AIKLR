from transformers import AutoTokenizer, AutoModel
from torch import nn
from torch.optim.lr_scheduler import StepLR

from new_architecture.AMRInfusion import AMRInfusion
from new_architecture.AMRInfusionMul import AMRInfusionMul
from sentence_transformers import ExtendSentenceTransformer
from preprocess import generate_ref_edge
from sentence_transformers import InputExample

import torch
import json
import numpy as np
import os
import argparse
from tqdm import tqdm


def summary_model(model):
    num_para = 0
    for p in model.parameters():
        if p.requires_grad:
            num_para += p.numel()
    print(num_para)
    # print(model)
    return 1


def prepare_text_input(question, article, text_tokenizer, model_type):
    tokenized_text = text_tokenizer(
        question,
        article,
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=512,
    )
    if model_type == "roberta":
        tokenized_text["token_type_ids"] = None
    # print(tokenized_text.keys())
    return tokenized_text


def check_amr_missing(line):
    err_flag = False
    if line["graph_ref1"]["amr_simple"] == "" or line["graph_ref2"]["amr_simple"] == "":
        err_flag = True
    return err_flag


def get_text_jp(filepath, qid):
    # filepath = "data_origin/coliee-2024/en_jp_train2024_v2.json"
    with open(filepath, "r") as f:
        data = json.load(f)

    question = ""
    article = ""
    for sample in data:
        if sample["id"] == qid:
            question = sample["jp_query"].replace("\n", " ")
            legal_text = ""
            for i, a in enumerate(sample["relevant_articles"]):
                article_text = a["article_content_jp"]
                # print(article_text[-1])
                # print(i)
                if article_text[-1] == ".":
                    legal_text += article_text + " "
                elif article_text[-1] != ".":
                    legal_text += article_text + ". "
                # print(legal_text)
            article = legal_text

    return question, article


def prepare_input(
    originpath,
    filepath,
    text_tokenizer,
    amr_tokenizer,
    model_type,
    device,
    text_lang="en",
):
    test_samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)

            # prepare text input
            # print(line)
            label = line["label"]

            qid = line["id"]
            if text_lang == "en":
                text_question = line["ref1"]
                text_article = line["ref2"]
            elif text_lang == "jp":
                text_question, text_article = get_text_jp(originpath, qid)
                # print(text_question, text_article)
            text_input = prepare_text_input(
                text_question, text_article, text_tokenizer, model_type
            )
            # text_input = [text_input_ids, text_attention_mask, text_token_type_ids]

            # print(text_question, text_article, text_input)
            # break

            # prepare amr input
            # max linearised graph length for AMRSim 64,128,256
            max_seq_length = 128
            err_flag = check_amr_missing(line)
            edge_index = None
            edge_type = None
            pos_ids = None
            if err_flag == False:
                edge_index, edge_type, pos_ids = generate_ref_edge(
                    line, amr_tokenizer, max_seq_length
                )
            amr_input = InputExample(
                texts=[
                    line["graph_ref1"]["amr_simple"],
                    line["graph_ref2"]["amr_simple"],
                ],
                edge_index=edge_index,
                edge_type=edge_type,
                pos_ids=pos_ids,
                err_flag=err_flag,
            )

            if label == 0:
                label = [[1], [0]]
            else:
                label = [[0], [1]]
            label = torch.tensor(label, dtype=torch.float32).view(-1, 2).to(device)

            # print(inp_example.texts)
            # print(inp_example.edge_index, inp_example.edge_type, inp_example.pos_ids)
            test_samples.append([qid, text_input, amr_input, label])
    return test_samples


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
        for i in out:
            f.write(json.dumps(i))
            f.write("\n")
    f.close()
    return 1


def train_one_epoch(model, train_ds, optimizer, scheduler):
    model.train()
    epoch_loss = 0

    for qid, text_input, amr_input, label in train_ds:
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
        for qid, text_input, amr_input, label in test_ds:
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


def train_multiple_epoch(model, train_ds, test_ds, num_eps, learning_rate, log_path, save_path):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer=optimizer, step_size=3, gamma=0.5)
    train_losses = []
    eval_losses = []
    eval_accs = []
    eval_logs = []
    
    for epoch in tqdm(range(num_eps)):
        checkpoint_path = save_path
        train_losses.append(train_one_epoch(model, train_ds, optimizer, scheduler))
        eval_acc, eval_loss, qids, probs, preds, labels = eval(model, test_ds)

        print("eval loss: ", eval_loss)
        print("eval acc: ", eval_acc)

        eval_losses.append(eval_loss)
        eval_accs.append(eval_acc)
        eval_logs.append([qids, probs, preds, labels])

        checkpoint_path += "/epoch_%d.pt" % epoch
        torch.save(model.state_dict(), checkpoint_path)

    best_epoch = np.argmax(eval_accs)
    print("best epoch: ", best_epoch)
    output_path = log_path
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
        choices=["only_amr", "only_text", "infusion"],
    )
    parser.add_argument(
        "-infusion_type", required=False, type=str, choices=["concat", "co_attn_res", "co_attn_res_v2"]
    )
    parser.add_argument("-text_lang", required=True, type=str, choices=["en", "jp"])
    parser.add_argument("-parser", required=True, type=str, choices=["gsp", "spring"])
    parser.add_argument("-data_year", required=True, type=int, choices=[2023, 2024])
    parser.add_argument("-log_path", required=False, type=str)
    parser.add_argument("-save_path", required=False, type=str)
    parser.add_argument(
        "-device", required=False, type=str, choices=["cuda", "cpu"], default="cuda"
    )
    parser.add_argument("-learning_rate", required=False, type=float, default=1e-3)
    parser.add_argument("-num_eps", required=False,type=int,default=10)
    args = parser.parse_args()
    
    # load pretrained AMR, BERT models and tokenizers
    amr_pretrained_path = "output/ct-wiki-bert"
    # amr_pretrained_path = "output/ct-amr-bert-finetuned"
    amr_model = ExtendSentenceTransformer(amr_pretrained_path)
    amr_tokenizers = amr_model.tokenizer

    # text_pretrained_path = "FacebookAI/roberta-large"
    # text_pretrained_path = "google-bert/bert-base-cased"
    # text_pretrained_path = "google-bert/bert-base-multilingual-cased"
    text_pretrained_path = "google-bert/bert-base-multilingual-uncased"

    text_model = AutoModel.from_pretrained(text_pretrained_path)
    text_tokenizer = AutoTokenizer.from_pretrained(text_pretrained_path)

    # model = only_text (using only text model embedddings)
    # mode = infusion (combine amr and text models embeddings)
    if text_pretrained_path.find("large") != -1:
        concat_emb_dim = 768 + 1024
        emb_dim = 1024
    else:
        concat_emb_dim = 768 * 2
        emb_dim = 768
    model = AMRInfusionMul(
        text_model=text_model,
        amr_model=amr_model,
        dropout=0.1,
        mode=args.model_type,
        concat_emb_dim=concat_emb_dim,
        emb_dim=emb_dim,
        infusion_type=args.infusion_type,
        batch_size=1,
        device=args.device,
    )
    model.to(args.device)

    (summary_model(model))
    if text_pretrained_path.find("roberta") != -1:
        model_type = "roberta"
    else:
        model_type = "bert"

    if args.data_year == 2024:
        if args.parser == "spring":
            train_path = "../data/spring/train_qa.json"
            test_path = "../data/spring/test_qa.json"
        elif args.parser == "gsp":
            train_path = "../data/amr-parser/en_train2024_qa_v2.json"
            test_path = "../data/amr-parser/en_test2024_qa_v2.json"
        raw_train_path = "../../data_origin/coliee-2024/en_jp_train2024_v2.json"
        raw_test_path = "../../data_origin/coliee-2024/en_jp_test2024_v2.json"

    elif args.data_year == 2023:
        if args.parser == "spring":
            train_path = "../data/spring/train2023_qa.json"
            test_path = "../data/spring/test2023_qa.json"
        elif args.parser == "gsp":
            train_path = "../data/amr-parser/en_train2023_qa_v2.json"
            test_path = "../data/amr-parser/en_test2023_qa_v2.json"
        raw_train_path = "../../data_origin/coliee-2023/en_jp_train2023_v2.json"
        raw_test_path = "../../data_origin/coliee-2023/en_jp_test2023_v2.json"

    train_ds = prepare_input(
        raw_train_path,
        train_path,
        text_tokenizer,
        amr_tokenizers,
        model_type=model_type,
        text_lang=args.text_lang,
        device=args.device,
    )
    test_ds = prepare_input(
        raw_test_path,
        test_path,
        text_tokenizer,
        amr_tokenizers,
        model_type=model_type,
        text_lang=args.text_lang,
        device=args.device,
    )

    print(len(train_ds), len(test_ds))
    print("mbert jp infusion SPRING 2024 amrsim")

    train_losses, test_losses, test_accs = train_multiple_epoch(
        model=model,
        train_ds=train_ds,
        test_ds=test_ds,
        num_eps=args.num_eps,
        learning_rate=args.learning_rate,
        log_path=args.log_path,
        save_path=args.save_path,
    )
    print("train_losses: ", train_losses)
    print("eval_losses: ", test_losses)
    print("test accuracy: ", test_accs)


if __name__ == "__main__":
    main()
