#!/usr/bin/env bash

export BERT_BASE_DIR=gs://test-bert-konok/cased_L-12_H-768_A-12

  python BERT_NER.py\
    --task_name="NER"  \
    --do_lower_case=False \
    --use_tpu=True \
    --tpu_name=$TPU_NAME \
    --crf=True \
    --do_train=True   \
    --do_eval=True   \
    --do_predict=True \
    --data_dir=data   \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128   \
    --train_batch_size=32   \
    --learning_rate=2e-5   \
    --num_train_epochs=4.0   \
    --output_dir=gs://test-bert-konok/output-for-bertner/


perl conlleval.pl -d '\t' < ./output/result_dir/label_test.txt
