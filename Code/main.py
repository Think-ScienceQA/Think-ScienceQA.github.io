import numpy as np
import torch
import os
import re
import json
import argparse
import random
from transformers import T5Tokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    T5ForConditionalGeneration
from model import T5ForConditionalGeneration, T5ForMultimodalGeneration
from utils_data import img_shape, load_data_std, load_data_img, ScienceQADatasetStd, ScienceQADatasetImg, ScienceQADatasetIterator
from utils_prompt import *
#from utils_prompt_pre import *
from utils_evaluate import get_scores
from rich.table import Column, Table
from rich import box
from rich.console import Console

console = Console(record=True)
from torch import cuda
import nltk
#nltk.download('punkt')
import evaluate

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--model', type=str, default='allenai/unifiedqa-t5-base')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--input_len', type=int, default=512)
    parser.add_argument('--output_len', type=int, default=64)
    parser.add_argument('--eval_bs', type=int, default=16)
    parser.add_argument('--eval_acc', type=int, default=None, help='evaluate accumulation step')
    parser.add_argument('--train_split', type=str, default='train', choices=['train', 'trainval', 'minitrain'])
    parser.add_argument('--val_split', type=str, default='val', choices=['test', 'val', 'minival'])
    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'minitest'])
    parser.add_argument('--freeze_unet', action= 'store_true')
    parser.add_argument('--freeze_vae',  action= 'store_true')
    parser.add_argument('--use_generate', action='store_true', help='only for baseline to improve inference speed')
    parser.add_argument('--final_eval', action='store_true', help='only evaluate the model at the final epoch')
    parser.add_argument('--user_msg', type=str, default="baseline", help='experiment type in the save_dir')
    parser.add_argument('--img_type', type=str, default=None, choices=['detr', 'clip', 'resnet', 'png'],
                        help='type of image features')
    parser.add_argument('--eval_le', type=str, default=None, help='generated rationale for the dev set')
    parser.add_argument('--test_le', type=str, default=None, help='generated rationale for the test set')
    parser.add_argument('--evaluate_dir', type=str, default=None, help='the directory of model for evaluation')
    parser.add_argument('--caption_file', type=str, default='data/captions.json')
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--prompt_format', type=str, default='QCM-A', help='prompt format template',
                        choices=['QCM-A', 'QCM-LE', 'QCMG-A', 'QCM-LEA', 'QCM-ALE','Q-E'])
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'])
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to the checkpoint to load')
    parser.add_argument('--use_newpre', action='store_true', help='use 240000 pretrain or not')
    parser.add_argument('--freeze_encoder', action='store_true', help=' ')
    parser.add_argument('--freeze_decoder', action='store_true', help=' ')
    parser.add_argument('--vot_num', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()
    return args


def T5Trainer(
        dataframe, args,
):
    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    if args.evaluate_dir is not None:
        args.model = args.evaluate_dir

    tokenizer = T5Tokenizer.from_pretrained(args.model)

    console.log(f"""[Model]: Loading {args.model}...\n""")
    console.log(f"[Data]: Reading data...\n")
    problems = dataframe['problems']
    qids = dataframe['qids']
    train_qids = qids['train']
    test_qids = qids['test']
    #test_qids = ['2183248']
    val_qids = qids['val']
    #val_qids = ['948865']

    if args.evaluate_dir is not None:
        save_dir = args.evaluate_dir
    else:
        model_name = args.model.replace("/", "-")
        gpu_count = torch.cuda.device_count()
        #save_dir = "answer_test_how_is_20"
        #ave_dir = "answer_20_frozen_semantic_model_PPDM_63"
        #save_dir = "/data/mm-cot-main/models/frozen_semantic_model_llavacot/"
        save_dir = f"{args.output_dir}/{args.user_msg}_{model_name}_{args.img_type}_{args.prompt_format}_lr{args.lr}_bs{args.bs * gpu_count}_op{args.output_len}_ep{args.epoch}_mc10_PS-ft_0310"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    padding_idx = tokenizer._convert_token_to_id(tokenizer.pad_token)
    if args.img_type is not None:
        patch_size = img_shape[args.img_type]
        if args.load_checkpoint is not None:
            # 创建模型
            model = T5ForMultimodalGeneration.from_pretrained(args.load_checkpoint, patch_size=patch_size, padding_idx=padding_idx, save_dir=save_dir,vot_num=args.vot_num,alpha=args.alpha, freeze_unet=args.freeze_unet, freeze_vae=args.freeze_vae,freeze_encoder=args.freeze_encoder,freeze_decoder=args.freeze_decoder,ignore_mismatched_sizes=True)

        else:
            model = T5ForMultimodalGeneration.from_pretrained(args.model, patch_size=patch_size, padding_idx=padding_idx,
                                                          save_dir=save_dir, vot_num=args.vot_num,alpha=args.alpha,freeze_unet=args.freeze_unet, freeze_vae=args.freeze_vae,freeze_encoder=args.freeze_encoder,freeze_decoder=args.freeze_decoder)
        name_maps = dataframe['name_maps']
        image_features = dataframe['image_features']
        train_set = ScienceQADatasetImg(
                problems,
                train_qids,
                name_maps,
                tokenizer,
                args.input_len,
                args.output_len,
                args,
                image_features,
            )
        eval_set = ScienceQADatasetImg(
            problems,
            val_qids,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features,
            args.eval_le,
        )
        test_set = ScienceQADatasetImg(
            problems,
            test_qids,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features,
            args.test_le,
        )
    else:
        model = T5ForConditionalGeneration.from_pretrained(args.model)
        train_set = ScienceQADatasetStd(
            problems,
            train_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
        )
        eval_set = ScienceQADatasetStd(
            problems,
            val_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            args.eval_le,
        )

        test_set = ScienceQADatasetStd(
            problems,
            test_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            args.test_le,
        )

    datacollator = DataCollatorForSeq2Seq(tokenizer)

    print("model parameters: ", model.num_parameters())
    #num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print("Number of trainable parameters: ", num_trainable_parameters)

    def extract_ans(ans):
        ans = ans.decode("utf-8") if isinstance(ans, bytes) else str(ans)  # 强制将ans转换为字符串

        pattern = re.compile(r'The answer is \(([A-Z])\)')
        res = pattern.findall(ans)

        if len(res) == 1:
            answer = res[0]  # 'A', 'B', ...
        else:
            answer = "FAILED"
        return answer

        # accuracy for answer inference


    # rougel for rationale generation
    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    def compute_metrics_acc(eval_preds):
        if args.use_generate:
            preds, targets = eval_preds.predictions, eval_preds.label_ids
        else:
            preds = eval_preds.predictions[0]
            targets = eval_preds.label_ids
            preds = preds.argmax(axis=2)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        correct = 0
        assert len(preds) == len(targets)
        for idx, pred in enumerate(preds):
            reference = targets[idx]
            reference = extract_ans(reference)
            extract_pred = extract_ans(pred)
            best_option = extract_pred
            if reference == best_option:
                correct += 1
        return {'accuracy': 1.0 * correct / len(targets)}

    def compute_metrics_rougel(eval_preds):
        if args.use_generate:
            preds, targets = eval_preds.predictions, eval_preds.label_ids
        else:
            preds = eval_preds.predictions[0]
            targets = eval_preds.label_ids
            preds = preds.argmax(axis=2)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        decoded_preds, decoded_labels = postprocess_text(preds, targets)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result



    # only use the last model for evaluation to save time
    if args.final_eval:
        training_args = Seq2SeqTrainingArguments(
            save_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=False,
            evaluation_strategy="no",
            logging_strategy="steps",
            save_strategy="epoch",
            save_total_limit=2,
            learning_rate=args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            predict_with_generate=args.use_generate,
            report_to="none",
        )
    # evaluate at each epoch
    else:
        training_args = Seq2SeqTrainingArguments(
            save_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=True,
            evaluation_strategy="epoch",
            logging_strategy="steps",
            save_strategy="epoch",
            save_total_limit=2,
            learning_rate=args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            metric_for_best_model="accuracy" if args.prompt_format != "QCM-LE" else "rougeL",
            predict_with_generate=args.use_generate,
            load_best_model_at_end=True,
            report_to="none",
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=datacollator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_acc if args.prompt_format != "QCM-LE" else compute_metrics_rougel
    )


    if args.evaluate_dir is None:
        trainer.train()
        trainer.save_model(save_dir)


    #batch_size = 200
    test_set_iterator = ScienceQADatasetIterator(dataset=test_set, batch_size=32)

    eval_metrics = {}
    for batch in test_set_iterator:
        batch_metrics = trainer.evaluate(eval_dataset=batch)
        for key, value in batch_metrics.items():
            if key not in eval_metrics:
                eval_metrics[key] = []
            eval_metrics[key].append(value)

    for key, value in eval_metrics.items():
        eval_metrics[key] = sum(value) / len(value)

    metrics = eval_metrics
    #metrics = trainer.evaluate(eval_dataset=test_set)
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)

     # Set batch size for prediction
    batch_size = 32  # Adjust this based on your available memory

    # Initialize the dataset iterator
    test_set_iterator = ScienceQADatasetIterator(dataset=test_set, batch_size=batch_size)

    # Initialize empty lists to store predictions and labels
    all_preds = []
    all_targets = []

    # Iterate through the test dataset using the iterator
    for batch in test_set_iterator:
        # Predict on the current batch
        predict_results = trainer.predict(test_dataset=batch, max_length=args.output_len)

        if args.use_generate:
            preds, targets = predict_results.predictions, predict_results.label_ids
        else:
            preds = predict_results.predictions[0]
            targets = predict_results.label_ids
            preds = preds.argmax(axis=2)

        # Append the predictions and labels to the respective lists
        all_preds.extend(preds)
        all_targets.extend(targets)

    # Decode the predictions and labels
    preds_decoded = tokenizer.batch_decode(
        all_preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    targets_decoded = tokenizer.batch_decode(
        all_targets, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    # Now, you can work with preds_decoded and targets_decoded as needed

    # Process the decoded predictions as before
    results_ans = {}
    results_rationale = {}
    results_reference = {}

    num_fail = 0
    for idx, qid in enumerate(test_qids):
        pred = preds_decoded[int(idx)]
        ref = targets_decoded[int(idx)]
        extract_pred = extract_ans(pred)
        if extract_pred != "FAILED":
            if extract_pred in args.options:
                extract_pred = args.options.index(extract_pred)
            else:
                extract_pred = random.choice(range(0, len(args.options)))
        else:
            num_fail += 1
            extract_pred = random.choice(range(len(args.options)))  # random choose one option
        results_ans[str(qid)] = extract_pred
        results_rationale[str(qid)] = pred
        results_reference[str(qid)] = ref

    scores = get_scores(results_ans, results_rationale, results_reference,
                        os.path.join(args.data_root, "scienceqa/problems.json"))
    preds_decoded = [pred.strip() for pred in preds_decoded]
    output_data = {
        "num_fail": num_fail,
        "scores": scores,
        "preds": preds_decoded,
        "labels": targets_decoded
    }
    output_prediction_file = os.path.join(save_dir, "predictions_ans_test.json")
    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(output_data, indent=4))

    # generate the rationale for the eval set
    if args.prompt_format == "QCM-LE":
        torch.cuda.empty_cache()
        del predict_results, preds_decoded, targets_decoded

        # Set batch size for prediction
        batch_size = 32  # Adjust this based on your available memory

        # Initialize the dataset iterator
        test_set_iterator = ScienceQADatasetIterator(dataset=eval_set, batch_size=batch_size)

        # Initialize empty lists to store predictions and labels
        all_preds = []
        all_targets = []

        # Iterate through the test dataset using the iterator
        for batch in test_set_iterator:
            # Predict on the current batch
            predict_results = trainer.predict(test_dataset=batch, max_length=args.output_len)

            if args.use_generate:
                preds, targets = predict_results.predictions, predict_results.label_ids
            else:
                preds = predict_results.predictions[0]
                targets = predict_results.label_ids
                preds = preds.argmax(axis=2)

            # Append the predictions and labels to the respective lists
            all_preds.extend(preds)
            all_targets.extend(targets)

        # Decode the predictions and labels
        preds_decoded = tokenizer.batch_decode(
            all_preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        targets_decoded = tokenizer.batch_decode(
            all_targets, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        preds_decoded = [pred.strip() for pred in preds_decoded]
        output_data = {"preds": preds_decoded,
                       "labels": targets_decoded}
        output_prediction_file = os.path.join(save_dir, "predictions_ans_eval.json")
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(output_data, indent=4))


if __name__ == '__main__':

    # training logger to log training progress
    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )

    args = parse_args()
    print("args", args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.img_type is not None:
        problems, qids, name_maps, image_features = load_data_img(args)  # probelms, test question ids, shot example ids
        dataframe = {'problems': problems, 'qids': qids, 'name_maps': name_maps, 'image_features': image_features}
    else:
        problems, qids = load_data_std(args)  # probelms, test question ids, shot example ids
        dataframe = {'problems': problems, 'qids': qids}


    T5Trainer(dataframe=dataframe, args=args)