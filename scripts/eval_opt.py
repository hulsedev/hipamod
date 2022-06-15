import json
import torch
import numpy as np

from torch.utils.data import DataLoader
from datasets import load_dataset
from evaluate import load
from transformers import GPT2Tokenizer, OPTModel


def main():
    # see https://huggingface.co/docs/transformers/v4.19.4/en/training
    # use the tokenizer refered to by OPT paper
    tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")
    model = OPTModel.from_pretrained("facebook/opt-350m")

    # as of end of May 2022, no public evaluation setup was provided by Meta AI
    squad_dataset = load_dataset("squad")
    print("Loaded squad dataset", squad_dataset)

    def preprocess_questions(samples):
        questions = [q.strip() for q in samples["question"]]
        inputs = tokenizer(
            questions,
            samples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = samples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if (
                offset[context_start][0] > end_char
                or offset[context_end][1] < start_char
            ):
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    tokenized_dataset = squad_dataset.map(preprocess_questions, batched=True)
    tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(1000))
    eval_dataloader = DataLoader(eval_dataset, batch_size=8)

    # compute metric for torch dataset
    squad_metric = load("squad")
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        squad_metric.add_batch(predictions=predictions, references=batch["labels"])

    results = squad_metric.compute()
    print(
        "obtained results for squad dataset & opt-350m:", json.dumps(results, indent=4)
    )


if __name__ == "__main__":
    main()
