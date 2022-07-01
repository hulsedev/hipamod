from pathlib import Path
import logging
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from transformers import (
    AlbertForMaskedLM,
    AutoConfig,
    AutoTokenizer,
    SquadV1Processor,
    squad_convert_examples_to_features,
)

logger = logging.getLogger(__name__)


model_type = "masked_bert"  # TODO: check if should implement something new for albert
model_name_or_path = model_ckpt = "qwant/fralbert-base"
output_dir = Path("latency/pilot/qwant/distil/")
data_dir = "latency/pilot/qwant/data/fquad/"  # TODO: check if need to download dataset
eval_file = "valid.json"
train_file = "train.jsn"
version_2_with_negative = False  # TODO: is this correct?
null_score_diff_threshold = 0.0
max_seq_length = 384
max_query_length = 64
doc_stride = 128
max_seq_length = 64
do_train, do_eval = True, True
evaluate_during_training = True
do_lower_case = True
per_gpu_train_batch_size = 16
per_gpu_train_batch_size = 16
learning_rate = 3e-5
mask_scores_learning_rate = 1e-2
initial_threshold = 0.0  # TODO: is correct? other default was 1.0
final_threshold = 0.1  # TODO: is correct? other default was 0.7
initial_warmup = 1
final_warmup = 2
pruning_method = "sigmoied_threshold"
mask_init = "constant"
mask_scale = 0.0
regularization = "l1"
final_lambda = 400.0  # TODO: is correct? default was 0
global_topk = False
global_topk_frequency_compute = 25

# for distillation, TODO: pre-distil fine-tune the teacher on fquad
teacher_type = model_type  # TODO: is correct? assuming will reuse same arch
teacher_name_or_path = model_ckpt
alpha_ce = 0.5
alpha_distil = 0.5

temperature = 2.0
gradient_accumulation_steps = 1
weight_decay = 0.0
adam_epsilon = 1e-8
max_grad_norm = 1.0
num_train_epochs = 10
max_steps = -1
warmup_steps = 5400
n_best_size = 20
max_answer_length = 30
lang_id = 2  # TODO: check which corresponds to "fr"
logging_steps = 500
save_steps = 500
seed = 42
local_rank = -1
fp16_opt_level = "O1"
threads = 1
fp16 = False
n_gpu = 0


def set_seed(n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def schedule_threshold(
    step: int,
    total_step: int,
    warmup_steps: int,
    initial_threshold: float,
    final_threshold: float,
    initial_warmup: int,
    final_warmup: int,
    final_lambda: float,
):
    if step <= initial_warmup * warmup_steps:
        threshold = initial_threshold
    elif step > (total_step - final_warmup * warmup_steps):
        threshold = final_threshold
    else:
        spars_warmup_steps = initial_warmup * warmup_steps
        spars_schedu_steps = (final_warmup + initial_warmup) * warmup_steps
        mul_coeff = 1 - (step - spars_warmup_steps) / (total_step - spars_schedu_steps)
        threshold = final_threshold + (initial_threshold - final_threshold) * (
            mul_coeff**3
        )
    regu_lambda = final_lambda * threshold / final_threshold
    return threshold, regu_lambda


def regularization(model: nn.Module, mode: str):
    regu, counter = 0, 0
    for name, param in model.named_parameters():
        if "mask_scores" in name:
            if mode == "l1":
                regu += torch.norm(torch.sigmoid(param), p=1) / param.numel()
            elif mode == "l0":
                regu += (
                    torch.sigmoid(param - 2 / 3 * np.log(0.1 / 1.1)).sum()
                    / param.numel()
                )
            else:
                ValueError("Don't know this mode.")
            counter += 1
    return regu / counter


def load_and_cache_examples(tokenizer, evaluate=False, output_examples=False):
    if local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    processor = SquadV1Processor()
    if evaluate:
        examples = processor.get_dev_examples(data_dir, filename=eval_file)
    else:
        examples = processor.get_train_examples(data_dir, filename=train_file)

    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=not evaluate,
        threads=threads,
        return_dataset="pt",
    )

    if local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features

    return dataset


def train(train_dataset, model, tokenizer, teacher=None):
    """Train the model"""
    if local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.output_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "mask_score" in n and p.requires_grad
            ],
            "lr": args.mask_scores_learning_rate,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "mask_score" not in n
                and p.requires_grad
                and not any(nd in n for nd in no_decay)
            ],
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "mask_score" not in n
                and p.requires_grad
                and any(nd in n for nd in no_decay)
            ],
            "lr": args.learning_rate,
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(
        os.path.join(args.model_name_or_path, "optimizer.pt")
    ) and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"))
        )
        scheduler.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"))
        )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    # Distillation
    if teacher is not None:
        logger.info("  Training with distillation")

    global_step = 1
    # Global TopK
    if args.global_topk:
        threshold_mem = None
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (
                len(train_dataloader) // args.gradient_accumulation_steps
            )
            steps_trained_in_current_epoch = global_step % (
                len(train_dataloader) // args.gradient_accumulation_steps
            )

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info(
                "  Will skip the first %d steps in the first epoch",
                steps_trained_in_current_epoch,
            )
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    # Added here for reproducibility
    set_seed(args)

    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            threshold, regu_lambda = schedule_threshold(
                step=global_step,
                total_step=t_total,
                warmup_steps=args.warmup_steps,
                final_threshold=args.final_threshold,
                initial_threshold=args.initial_threshold,
                final_warmup=args.final_warmup,
                initial_warmup=args.initial_warmup,
                final_lambda=args.final_lambda,
            )
            # Global TopK
            if args.global_topk:
                if threshold == 1.0:
                    threshold = -1e2  # Or an indefinitely low quantity
                else:
                    if (threshold_mem is None) or (
                        global_step % args.global_topk_frequency_compute == 0
                    ):
                        # Sort all the values to get the global topK
                        concat = torch.cat(
                            [
                                param.view(-1)
                                for name, param in model.named_parameters()
                                if "mask_scores" in name
                            ]
                        )
                        n = concat.numel()
                        kth = max(n - (int(n * threshold) + 1), 1)
                        threshold_mem = concat.kthvalue(kth).values.item()
                        threshold = threshold_mem
                    else:
                        threshold = threshold_mem
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                if args.version_2_with_negative:
                    inputs.update({"is_impossible": batch[7]})
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {
                            "langs": (
                                torch.ones(batch[0].shape, dtype=torch.int64)
                                * args.lang_id
                            ).to(args.device)
                        }
                    )

            if "masked" in args.model_type:
                inputs["threshold"] = threshold

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss, start_logits_stu, end_logits_stu = outputs

            # Distillation loss
            if teacher is not None:
                with torch.no_grad():
                    start_logits_tea, end_logits_tea = teacher(
                        input_ids=inputs["input_ids"],
                        token_type_ids=inputs["token_type_ids"],
                        attention_mask=inputs["attention_mask"],
                    )

                loss_start = nn.functional.kl_div(
                    input=nn.functional.log_softmax(
                        start_logits_stu / args.temperature, dim=-1
                    ),
                    target=nn.functional.softmax(
                        start_logits_tea / args.temperature, dim=-1
                    ),
                    reduction="batchmean",
                ) * (args.temperature**2)
                loss_end = nn.functional.kl_div(
                    input=nn.functional.log_softmax(
                        end_logits_stu / args.temperature, dim=-1
                    ),
                    target=nn.functional.softmax(
                        end_logits_tea / args.temperature, dim=-1
                    ),
                    reduction="batchmean",
                ) * (args.temperature**2)
                loss_logits = (loss_start + loss_end) / 2.0

                loss = args.alpha_distil * loss_logits + args.alpha_ce * loss

            # Regularization
            if args.regularization is not None:
                regu_ = regularization(model=model, mode=args.regularization)
                loss = loss + regu_lambda * regu_

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm
                    )
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    tb_writer.add_scalar("threshold", threshold, global_step)
                    for name, param in model.named_parameters():
                        if not param.requires_grad:
                            continue
                        tb_writer.add_scalar(
                            "parameter_mean/" + name, param.data.mean(), global_step
                        )
                        tb_writer.add_scalar(
                            "parameter_std/" + name, param.data.std(), global_step
                        )
                        tb_writer.add_scalar(
                            "parameter_min/" + name, param.data.min(), global_step
                        )
                        tb_writer.add_scalar(
                            "parameter_max/" + name, param.data.max(), global_step
                        )
                        if "pooler" in name:
                            continue
                        tb_writer.add_scalar(
                            "grad_mean/" + name, param.grad.data.mean(), global_step
                        )
                        tb_writer.add_scalar(
                            "grad_std/" + name, param.grad.data.std(), global_step
                        )
                        if args.regularization is not None and "mask_scores" in name:
                            if args.regularization == "l1":
                                perc = (
                                    torch.sigmoid(param) > threshold
                                ).sum().item() / param.numel()
                            elif args.regularization == "l0":
                                perc = (
                                    torch.sigmoid(param - 2 / 3 * np.log(0.1 / 1.1))
                                ).sum().item() / param.numel()
                            tb_writer.add_scalar(
                                "retained_weights_perc/" + name, perc, global_step
                            )

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                "eval_{}".format(key), value, global_step
                            )
                    learning_rate_scalar = scheduler.get_lr()
                    tb_writer.add_scalar("lr", learning_rate_scalar[0], global_step)
                    if len(learning_rate_scalar) > 1:
                        for idx, lr in enumerate(learning_rate_scalar[1:]):
                            tb_writer.add_scalar(f"lr/{idx+1}", lr, global_step)
                    tb_writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_step,
                    )
                    if teacher is not None:
                        tb_writer.add_scalar(
                            "loss/distil", loss_logits.item(), global_step
                        )
                    if args.regularization is not None:
                        tb_writer.add_scalar(
                            "loss/regularization", regu_.item(), global_step
                        )
                    if (teacher is not None) or (args.regularization is not None):
                        if (teacher is not None) and (args.regularization is not None):
                            tb_writer.add_scalar(
                                "loss/instant_ce",
                                (
                                    loss.item()
                                    - regu_lambda * regu_.item()
                                    - args.alpha_distil * loss_logits.item()
                                )
                                / args.alpha_ce,
                                global_step,
                            )
                        elif teacher is not None:
                            tb_writer.add_scalar(
                                "loss/instant_ce",
                                (loss.item() - args.alpha_distil * loss_logits.item())
                                / args.alpha_ce,
                                global_step,
                            )
                        else:
                            tb_writer.add_scalar(
                                "loss/instant_ce",
                                loss.item() - regu_lambda * regu_.item(),
                                global_step,
                            )
                    logging_loss = tr_loss

                # Save model checkpoint
                if (
                    args.local_rank in [-1, 0]
                    and args.save_steps > 0
                    and global_step % args.save_steps == 0
                ):
                    output_dir = os.path.join(
                        args.output_dir, "checkpoint-{}".format(global_step)
                    )
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(
                        optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                    )
                    torch.save(
                        scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                    )
                    logger.info(
                        "Saving optimizer and scheduler states to %s", output_dir
                    )

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(model, tokenizer, prefix=None):
    pass


def main():
    """
    Reproducing soft-movement pruning from the Hugging Face transformers research
    project directory. This method both prunes the model while fine-tuning it on
    the FQuAD QA dataset, and distils it. We strive for 3% sparsity (leaving the
    memory footprint, and inference speed untouched), and hope to make further
    progress once everything is implemented.
    
    see https://github.com/huggingface/transformers/tree/main/examples/research_projects/movement-pruning
    
    python examples/movement-pruning/masked_run_squad.py \
    --output_dir $SERIALIZATION_DIR \
    --data_dir $SQUAD_DATA \
    --train_file train-v1.1.json \
    --predict_file dev-v1.1.json \
    --do_train --do_eval --do_lower_case \
    --model_type masked_bert \
    --model_name_or_path bert-base-uncased \
    --per_gpu_train_batch_size 16 \
    --warmup_steps 5400 \
    --num_train_epochs 10 \
    --learning_rate 3e-5 --mask_scores_learning_rate 1e-2 \
    --initial_threshold 0 --final_threshold 0.1 \
    --initial_warmup 1 --final_warmup 2 \
    --pruning_method sigmoied_threshold --mask_init constant --mask_scale 0. \
    --regularization l1 --final_lambda 400.
    """

    if local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend="nccl")
        n_gpu = 1

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        local_rank,
        device,
        n_gpu,
        bool(local_rank != -1),
        fp16,
    )

    set_seed(n_gpu)

    # Load pretrained model and tokenizer
    if local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    config = AutoConfig.from_pretrained(
        model_ckpt,
        pruning_method=pruning_method,
        mask_init=mask_init,
        mask_scale=mask_scale,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_ckpt,
        do_lower_case=do_lower_case,
    )
    model = AlbertForMaskedLM.from_pretrained(
        model_ckpt,
        config=config,
    )

    if (
        teacher_type is not None and False
    ):  # TODO: implement further fine-tuning & distillation
        assert teacher_name_or_path is not None
        assert alpha_distil > 0.0
        assert alpha_distil + alpha_ce > 0.0

        # load teacher network
        teacher_config = AutoConfig.from_pretrained(model_ckpt)
        teacher = AlbertForMaskedLM.from_pretrained(
            model_ckpt,
            config=teacher_config,
        )
        teacher.to(device)
    else:
        teacher = None

    if local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(device)

    # Training
    if do_train:
        train_dataset = load_and_cache_examples(
            tokenizer, evaluate=False, output_examples=False
        )
        global_step, tr_loss = train(train_dataset, model, tokenizer, teacher=teacher)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if do_train and (local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", output_dir)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # TODO: format training args for saving
        # torch.save(args, os.path.join(output_dir, "training_args.bin"))

        model = AlbertForMaskedLM.from_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            output_dir, do_lower_case=do_lower_case
        )
        model.to(device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if do_eval and local_rank in [-1, 0]:
        if do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [output_dir]
        else:
            logger.info("Loading checkpoint %s for evaluation", model_name_or_path)
            checkpoints = [model_ckpt]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = AlbertForMaskedLM.from_pretrained(checkpoint)
            model.to(device)

            # Evaluate
            result = evaluate(model, tokenizer, prefix=global_step)

            result = dict(
                (k + ("_{}".format(global_step) if global_step else ""), v)
                for k, v in result.items()
            )
            results.update(result)

    logger.info("Results: {}".format(results))

    # TODO: check if need to add output for eval results
    # predict_file = list(filter(None, args.predict_file.split("/"))).pop()
    # if not os.path.exists(os.path.join(args.output_dir, predict_file)):
    #    os.makedirs(os.path.join(args.output_dir, predict_file))
    # output_eval_file = os.path.join(args.output_dir, predict_file, "eval_results.txt")
    # with open(output_eval_file, "w") as writer:
    #    for key in sorted(results.keys()):
    #        writer.write("%s = %s\n" % (key, str(results[key])))

    return results


if __name__ == "__main__":
    main()
