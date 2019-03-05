SEED=$1
CUDA=$2
SAMPLE=$3
CUDA_VISIBLE_DEVICES=$CUDA python run_classifier.py --task_name IMDB --do_train --do_eval --do_lower_case --data_dir /home/ubuntu/final-datasets/imdb/ --bert_model ~/pretrained-models/bert/imdb/ --max_seq_length 400 --train_batcsize 16 --learning_rate 2e-5 --num_train_epochs 3.0 --sample $SAMPLE --output_dir ./imdb_10000_$SEED --seed $SEED  --gradient_accumulation_steps 4
