from typing import Optional, Tuple
import numpy as np
import collections
import json
import os

from accelerate import Accelerator
from tqdm import tqdm
from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    default_data_collator,
    EvalPrediction,
    DataCollatorWithPadding,
)


def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
):
    """
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.
    """
    if len(predictions) != 2:
        raise ValueError(
            "`predictions` should be a tuple with two elements (start_logits, end_logits)."
        )
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(
            f"Got {len(predictions[0])} predictions and {len(features)} features."
        )

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get(
                "token_is_max_context", None
            )

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if (
                min_null_prediction is None
                or min_null_prediction["score"] > feature_null_score
            ):
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or len(offset_mapping[start_index]) < 2
                        or offset_mapping[end_index] is None
                        or len(offset_mapping[end_index]) < 2
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if (
                        token_is_max_context is not None
                        and not token_is_max_context.get(str(start_index), False)
                    ):
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (
                                offset_mapping[start_index][0],
                                offset_mapping[end_index][1],
                            ),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        if version_2_with_negative and min_null_prediction is not None:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(
            prelim_predictions, key=lambda x: x["score"], reverse=True
        )[:n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score.
        if (
            version_2_with_negative
            and min_null_prediction is not None
            and not any(p["offsets"] == (0, 0) for p in predictions)
        ):
            predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0 or (
            len(predictions) == 1 and predictions[0]["text"] == ""
        ):
            predictions.insert(
                0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}
            )

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            # Otherwise we first need to find the best non-empty prediction.
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # Then we compare to the null prediction using the threshold.
            score_diff = (
                null_score
                - best_non_null_pred["start_logit"]
                - best_non_null_pred["end_logit"]
            )
            scores_diff_json[example["id"]] = float(
                score_diff
            )  # To be JSON-serializable.
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [
            {
                k: (
                    float(v)
                    if isinstance(v, (np.float16, np.float32, np.float64))
                    else v
                )
                for k, v in pred.items()
            }
            for pred in predictions
        ]

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise EnvironmentError(f"{output_dir} is not a directory.")

        prediction_file = os.path.join(
            output_dir,
            "predictions.json" if prefix is None else f"{prefix}_predictions.json",
        )
        nbest_file = os.path.join(
            output_dir,
            "nbest_predictions.json"
            if prefix is None
            else f"{prefix}_nbest_predictions.json",
        )
        if version_2_with_negative:
            null_odds_file = os.path.join(
                output_dir,
                "null_odds.json" if prefix is None else f"{prefix}_null_odds.json",
            )

        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        if version_2_with_negative:
            with open(null_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions


def main(model_ckpt):
    # https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py
    dataset_name = "boolq"
    task_name = "question-answering"
    raw_datasets = load_dataset(dataset_name)

    accelerator = Accelerator()

    config = AutoConfig.from_pretrained(model_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, use_fast=True)
    print(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(model_ckpt, config=config)

    column_names = raw_datasets["validation"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[2]
    answer_column_name = "answers" if "answers" in column_names else column_names[1]

    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(384, tokenizer.model_max_length)

    def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [
            q.lstrip() for q in examples[question_column_name]
        ]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # TODO: double check parameters for evaluation with metaseq

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    eval_examples = raw_datasets["validation"]
    with accelerator.main_process_first():
        eval_dataset = eval_examples.map(
            prepare_validation_features,
            batched=True,
            remove_columns=column_names,
        )

    data_collator = DataCollatorWithPadding(tokenizer)

    # TODO: check arguments with boolq dataset + metaseq for reproducing experiments
    null_score_diff_threshold = 0.0
    version_2_with_negative = False
    n_best_size = 20
    max_answer_length = 30

    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=version_2_with_negative,
            n_best_size=n_best_size,
            max_answer_length=max_answer_length,
            null_score_diff_threshold=null_score_diff_threshold,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
                for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [
                {"id": k, "prediction_text": v} for k, v in predictions.items()
            ]

        references = [
            {"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples
        ]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = load_metric("boolq")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)