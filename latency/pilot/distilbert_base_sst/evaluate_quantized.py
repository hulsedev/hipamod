from pathlib import Path

import evaluate
import datasets
import transformers
import torch
from accelerate import Accelerator
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from torch.utils.data import DataLoader
from optimum.onnxruntime.modeling_ort import ORTModelForSequenceClassification
from transformers.models.distilbert.configuration_distilbert import DistilBertOnnxConfig


def main(model_filename):
    # using https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue_no_trainer.py
    accelerator = Accelerator()
    # datasets.utils.logging.set_verbosity_warning()
    # transformers.utils.logging.set_verbosity_info()

    task_name = "sst2"
    max_length = 128
    metric = evaluate.load("glue", task_name)
    raw_dataset = load_dataset("glue", task_name)
    is_regression = task_name == "stsb"
    if not is_regression:
        label_list = raw_dataset["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    model_ckpt = "distilbert-base-uncased-finetuned-sst-2-english"
    from_transformers = False
    config = AutoConfig.from_pretrained(
        model_ckpt, num_labels=num_labels, finetuning_task=task_name
    )
    if from_transformers:
        print("Loading default model from HF hub")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_ckpt,
            config=config,
        )
    else:
        print("Loading model from ONNX")
        model_dir = Path("latency/pilot/distilbert_base_sst/model/")
        if not model_dir.is_dir():
            model_dir.mkdir(parents=True)
        model = ORTModelForSequenceClassification.from_pretrained(
            model_dir, file_name=model_filename
        )

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    # use pre-defined map between label and ids
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            print(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {
                i: label_name_to_id[label_list[i]] for i in range(num_labels)
            }
        else:
            print(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    sentence1_key, sentence2_key = "sentence", None
    padding = False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *texts, padding=padding, max_length=max_length, truncation=True
        )

        if "label" in examples:
            if label_to_id is not None and False:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]

        return result

    # process dataset to extract labels + tokenize text
    with accelerator.main_process_first():
        processed_datasets = raw_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_dataset["train"].column_names,
            desc="Running tokenizer on dataset",
        )
    eval_dataset = processed_datasets["validation"]
    data_collator = DataCollatorWithPadding(tokenizer)

    # train_dataloader = DataLoader(
    #    train_dataset, shuffle=True, collate_fn=data_collator, batch_size=128
    # )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=128)

    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    # TODO: check if this statement is necessary
    if from_transformers and False:
        model.eval()

    samples_seen = 0
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = (
            outputs.logits.argmax(dim=-1)
            if not is_regression
            else outputs.logits.squeeze()
        )
        predictions, references = accelerator.gather((predictions, batch["labels"]))
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(eval_dataloader) - 1:
                predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                references = references[: len(eval_dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
        recall.add_batch(predictions=predictions, references=references)
        f1.add_batch(predictions=predictions, references=references)
        precision.add_batch(predictions=predictions, references=references)

    eval_metric = metric.compute()
    f1_metric = f1.compute()
    recall_metric = recall.compute()
    precision_metric = precision.compute()
    # print("Eval metric:", eval_metric)
    # print("Precision:", precision_metric)
    # print("Recall:", recall_metric)
    # print("F1:", f1_metric)

    return eval_metric | f1_metric | recall_metric | precision_metric


if __name__ == "__main__":
    main("model_quantized.onnx")
