#!/bin/bash

path_to_top_dir= # path to the top directory (directory of readme)

devices=$(echo -e "from gpuinfo import GPUInfo\nprint(GPUInfo.check_empty()[0])" | python)
nematus_home=$path_to_top_dir/codes/nematus
exp_name=iwslt/pt
src=de
tgt=en
data_dir=$path_to_top_dir/data/iwslt/iwslt-en-de
ood_dir=$path_to_top_dir/data/iwslt/domain_test_data
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
    --source_dataset $data_dir/pt/train.tok.tc.bpe.$src \
    --target_dataset $data_dir/pt/train.tok.tc.bpe.$tgt \
    --dictionaries $data_dir/train.tok.tc.bpe.both.special_tokens.json \
                   $data_dir/train.tok.tc.bpe.both.special_tokens.json \
    --save_freq 24000 \
    --valid_freq 4000 \
    --disp_freq 500 \
    --summary_freq 0 \
    --sample_freq 0 \
    --beam_freq 0 \
    --valid_source_dataset $data_dir/pt/dev.aux_123_base.$src \
    --valid_target_dataset $data_dir/pt/dev.aux_123_base.$tgt \
    --valid_batch_size 128 \
    --valid_token_batch_size 4096 \
    --valid_script $script_dir/validate.sh \
    --batch_size 64 \
    --token_batch_size 12288 \
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
    --maxlen 300 \
    --beam_size 4 \
    --translation_maxlen 300 \
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
    --max_tokens_per_device 16384 \


# evaluate
MOSES=$path_to_top_dir/codes/mosesdecoder
domains=(medical it law)
postprocess_script=$script_dir/extract_targeted_seq.py

# decode
for task in aux_123_base aux_321_base aux_213_base aux_231_base aux_132_base aux_312_base;do
    for length in 750;do
        for beam in 5;do
            for i in model.best-valid-script; do
                for domain in "${domains[@]}";do
                    CUDA_VISIBLE_DEVICES=$devices python3 $nematus_home/nematus/translate.py \
                        -m $working_dir/$i \
                        -i $ood_dir/$domain/test.tok.tc.bpe.$src.$task \
                        -o $working_dir/$i.$domain.test.tok.tc.bpe.$tgt.$task.beam$beam.length$length \
                        -k $beam \
                        -n 0.6 \
                        -b 50 \
                        --translation_maxlen $length

                    echo "detokenized BLEU of target seq of $i on $domain of beam $beam length $length of task $task:"
                    cat $working_dir/$i.$domain.test.tok.tc.bpe.$tgt.$task.beam$beam.length$length | python $postprocess_script tgt | sed -r 's/\@\@ //g' | $MOSES/scripts/recaser/detruecase.perl | $MOSES/scripts/tokenizer/detokenizer.perl -l $tgt | sacrebleu $ood_dir/$domain/test.$tgt -m bleu -sh -f text
                done

                CUDA_VISIBLE_DEVICES=$devices python3 $nematus_home/nematus/translate.py \
                    -m $working_dir/$i \
                    -i $data_dir/test.tok.tc.bpe.$src.$task \
                    -o $working_dir/$i.test.tok.tc.bpe.$tgt.$task.beam$beam.length$length \
                    -k $beam \
                    -n 0.6 \
                    -b 50 \
                    --translation_maxlen $length

                echo "detokenized BLEU of target seq of $i of beam $beam length $length of task $task:"
                cat $working_dir/$i.test.tok.tc.bpe.$tgt.$task.beam$beam.length$length | python $postprocess_script tgt | sed -r 's/\@\@ //g' | $MOSES/scripts/recaser/detruecase.perl | $MOSES/scripts/tokenizer/detokenizer.perl -l $tgt | sacrebleu $data_dir/test.$tgt -m bleu -sh -f text
            done
        done
    done
done
