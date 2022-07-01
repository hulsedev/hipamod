python latency/pilot/qwant/finetune_fquad.py \
--model_name_or_path qwant/fralbert-base \
--dataset_name squad \
--do_train \
--do_eval \
--per_device_train_batch_size 12 \
--learning_rate 3e-5 \
--num_train_epochs 2 \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir latency/pilot/qwant/tmp/debug_squad/

#--train_file latency/pilot/qwant/data/fquad/train.json \
#--validation_file latency/pilot/qwant/data/fquad/valid.json \
