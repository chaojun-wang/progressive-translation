#!/bin/bash

path_to_top_dir= # path to the top directory (directory of readme)

devices=$(echo -e "from gpuinfo import GPUInfo\nprint(GPUInfo.check_empty()[0])" | python)
nematus_home=$path_to_top_dir/codes/nematus
exp_name=iwslt/baseline
src=de
tgt=en
data_dir=$path_to_top_dir/data/iwslt/iwslt-en-de
script_dir=$path_to_top_dir/scripts/$exp_name
working_dir=$path_to_top_dir/models/$exp_name

echo $$
echo "CUDA_VISIBLE_DEVICES:" $devices
hostname
gpustat
nvidia-smi
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits

mkdir -p $working_dir
chmod 777 $script_dir/validate.sh

CUDA_VISIBLE_DEVICES=$devices python3 $nematus_home/nematus/train.py \
    --source_dataset $data_dir/train.tok.tc.bpe.$src \
    --target_dataset $data_dir/train.tok.tc.bpe.$tgt \
    --dictionaries $data_dir/train.tok.tc.bpe.both.json \
                   $data_dir/train.tok.tc.bpe.both.json \
    --save_freq 24000 \
    --valid_freq 4000 \
    --disp_freq 500 \
    --summary_freq 0 \
    --sample_freq 0 \
    --beam_freq 0 \
    --valid_source_dataset $data_dir/dev.tok.tc.bpe.$src \
    --valid_target_dataset $data_dir/dev.tok.tc.bpe.$tgt \
    --valid_batch_size 64 \
    --valid_token_batch_size 4096 \
    --valid_script $script_dir/validate.sh \
    --batch_size 64 \
    --token_batch_size 4096 \
    --model $working_dir/model \
    --reload latest_checkpoint \
    --model_type transformer \
    --embedding_size 512 \
    --state_size 512 \
    --tie_decoder_embeddings \
    --tie_encoder_decoder_embeddings \
    --loss_function per-token-cross-entropy \
    --label_smoothing 0.1 \
    --optimizer adam \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-09 \
    --learning_schedule transformer \
    --warmup_steps 4000 \
    --maxlen 100 \
    --beam_size 4 \
    --translation_maxlen 100 \
    --normalization_alpha 0.6 \
    --transformer_enc_depth 6 \
    --transformer_dec_depth 6 \
    --transformer_ffn_hidden_size 1024 \
    --transformer_num_heads 4 \
    --transformer_dropout_embeddings 0.3 \
    --transformer_dropout_residual 0.3 \
    --transformer_dropout_relu 0.3 \
    --transformer_dropout_attn 0.3 \
    --patience 10 \
    --clip_c 1 \

ood_data_dir=$path_to_top_dir/data/iwslt/domain_test_data
domains=(medical it law)
MOSES=$path_to_top_dir/codes/mosesdecoder
# evaluation
for i in model.best-valid-script; do
    for beam in 5;do
        for domain in "${domains[@]}";do
            CUDA_VISIBLE_DEVICES=$devices python3 $nematus_home/nematus/translate.py \
                -m $working_dir/$i \
                -i $ood_data_dir/$domain/test.tok.tc.bpe.$src \
                -o $working_dir/$i.$domain.test.tok.tc.bpe.$tgt.beam$beam \
                -k $beam \
                -n 0.6 \
                -b 50 \
                --translation_maxlen 200

            echo "detokenized BLEU of $i on $domain with beam $beam:"
            cat $working_dir/$i.$domain.test.tok.tc.bpe.$tgt.beam$beam | sed -r 's/\@\@ //g' | $MOSES/scripts/recaser/detruecase.perl | $MOSES/scripts/tokenizer/detokenizer.perl -l $tgt | sacrebleu $ood_data_dir/$domain/test.$tgt -m bleu -sh -f text
        done
    done
done

for i in model.best-valid-script; do
    for beam in 5;do
        CUDA_VISIBLE_DEVICES=$devices python3 $nematus_home/nematus/translate.py \
            -m $working_dir/$i \
            -i $data_dir/test.tok.tc.bpe.$src \
            -o $working_dir/$i.test.tok.tc.bpe.$tgt.beam$beam \
            -k $beam \
            -n 0.6 \
            -b 50 \
            --translation_maxlen 200

        echo "detokenized BLEU of $i with beam $beam:"
        cat $working_dir/$i.test.tok.tc.bpe.$tgt.beam$beam | sed -r 's/\@\@ //g' | $MOSES/scripts/recaser/detruecase.perl | $MOSES/scripts/tokenizer/detokenizer.perl -l $tgt | sacrebleu $data_dir/test.$tgt -m bleu -sh -f text
    done
done
