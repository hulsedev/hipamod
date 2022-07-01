python3 latency/pilot/qwant/finetune_fquad.py \
--model_name_or_path qwant/fralbert-base \
--do_train \
--do_eval \
--per_device_train_batch_size 12 \
--learning_rate 3e-5 \
--num_train_epochs 10 \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir latency/pilot/qwant/tmp/debug_squad/ \
--overwrite_output_dir \
--dataset_loading_script latency/pilot/qwant/fquad.py \

#--max_train_samples 1
#--max_eval_samples 1
#--dataset_name squad \
#--train_file latency/pilot/qwant/data/fquad/train.json \
#--validation_file latency/pilot/qwant/data/fquad/valid.json \
