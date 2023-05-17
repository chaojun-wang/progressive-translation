#!/bin/sh
# Distributed under MIT license

path_to_top_dir= # path to the top directory (directory of readme)

translations=$1
tgt=en
src=de
nematus_home=$path_to_top_dir/codes/nematus
data_dir=$path_to_top_dir/data/iwslt/iwslt-en-de
MOSES=$path_to_top_dir/codes/mosesdecoder
postprocess_script=$path_to_top_dir/scripts/iwslt/pt/extract_targeted_seq.py

cat $translations | python $postprocess_script tgt | sed -r 's/\@\@ //g' | $MOSES/scripts/recaser/detruecase.perl | $MOSES/scripts/tokenizer/detokenizer.perl -l $tgt | \
    $nematus_home/data/multi-bleu-detok.perl $data_dir/dev.$tgt | \
    cut -f 3 -d ' ' | \
    cut -f 1 -d ','
