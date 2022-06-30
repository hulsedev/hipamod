import evaluate
import datasets
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig
from tqdm import tqdm

from latency.hugging_face import infinity


def main():
    device = "cpu"
    model_ckpt = "distilbert-base-uncased-finetuned-sst-2-english"
    task = "sequence-classification"
    glue_task = "sst2"
    max_length = 128

    accelerator = Accelerator()

    accuracy = evaluate.load("glue", glue_task)
    precision = evaluate.load("precision")
    f1 = evaluate.load("f1")
    recall = evaluate.load("recall")

    is_regression = False
    raw_dataset = datasets.load_dataset("glue", glue_task)
    label_list = raw_dataset["train"].features["label"].names
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(
        model_ckpt, num_labels=num_labels, finetuning_task=glue_task
    )
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    label_to_id = None
    if (
        config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in config.label2id.items()}
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
        config.label2id = label_to_id
        config.id2label = {id: label for label, id in config.label2id.items()}
    elif not is_regression:
        config.label2id = {l: i for i, l in enumerate(label_list)}
        config.id2label = {id: label for label, id in config.label2id.items()}

    padding = False

    def preprocess_function(examples):
        return {"inputs": examples["sentence"], "labels": examples["label"]}

    with accelerator.main_process_first():
        processed_datasets = raw_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_dataset["train"].column_names,
        )
    eval_dataset = processed_datasets["validation"]
    eval_dataloader = DataLoader(eval_dataset, batch_size=32)
    for step, batch in tqdm(
        enumerate(eval_dataloader), total=int(len(eval_dataset) / 32)
    ):
        resp, _ = infinity.make_request(batch["inputs"], model_ckpt, device, task)
        predictions = [r.get("class") for r in resp]
        predictions, references = accelerator.gather((predictions, batch["labels"]))

        accuracy.add_batch(predictions=predictions, references=references)
        precision.add_batch(predictions=predictions, references=references)
        f1.add_batch(predictions=predictions, references=references)
        recall.add_batch(predictions=predictions, references=references)

    accuracy_metric = accuracy.compute()
    f1_metric = f1.compute()
    precision_metric = precision.compute()
    recall_metric = recall.compute()
    # print("Accuracy:", accuracy_metric)
    # print("F1:", f1_metric)
    # print("Precision:", precision_metric)
    # print("Recall:", recall_metric)

    return accuracy_metric | f1_metric | recall_metric | precision_metric


if __name__ == "__main__":
    main()
