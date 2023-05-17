import os
import sys
path_to_top_dir = # path to the top directory (directory of readme)
sys.path.append(f"{path_to_top_dir}/scripts")
from data_generation import shell, generate_lexical_map_seq_dicbased, apply_BPE, return_number_of_rows_of_corpus, cat_specific_line, generate_mono_data, generate_aux_seq_data, generate_aux_seq_src_data, logging

bpe_threshold=1
in_domain_dir = f"{path_to_top_dir}/data/iwslt/iwslt-en-de"
id_bpe_model_dir = f"{in_domain_dir}/model"
ood_domain_dir = f"{path_to_top_dir}/data/iwslt/domain_test_data"
domains = ["medical", "law", "it"]
bpeDir = f"{path_to_top_dir}/codes/subword-nmt/subword_nmt"

# ------------template 1 start----------------------------------------- #
logging.info("Start")
logging.info(f"pid is: {os.getpid()}")

store_data_dir = f"{in_domain_dir}/pt"
shell(f"mkdir -p {store_data_dir}")
tgt = "en"
src = "de"

# src, tgt
shell(f"cat {in_domain_dir}/train.tok.tc.bpe.{src} {in_domain_dir}/dev.tok.tc.bpe.{src} {in_domain_dir}/test.tok.tc.bpe.{src} > {store_data_dir}/src.bpe")
shell(f"cat {in_domain_dir}/train.tok.tc.{src} {in_domain_dir}/dev.tok.tc.{src} {in_domain_dir}/test.tok.tc.{src} > {store_data_dir}/src")
shell(f"cat {in_domain_dir}/train.tok.tc.bpe.{tgt} {in_domain_dir}/dev.tok.tc.bpe.{tgt} {in_domain_dir}/test.tok.tc.bpe.{tgt} > {store_data_dir}/tgt.bpe")
f_src = f"{store_data_dir}/src.bpe"
f_src_unbpenized = f"{store_data_dir}/src"
f_tgt = f"{store_data_dir}/tgt.bpe"

# dictionary
dict_file = f"{in_domain_dir}/alignment/intersection/model/lex.e2f.json"

# lex
dict_file = dict_file
source_corpus = f_src_unbpenized
lex_seq = f"{store_data_dir}/lex" 
generate_lexical_map_seq_dicbased(source_corpus, dict_file, lex_seq, lower_src=False, fix_lexical_map=True)

bpe_model_dir = id_bpe_model_dir
tgt_voc_dir = f"{bpe_model_dir}/vocab.{tgt}"
param = f"{bpe_model_dir}/{src}{tgt}.bpe"

output_lex_path = f"{store_data_dir}/lex.bpe" 
apply_BPE(bpeDir, lex_seq, tgt_voc_dir, bpe_threshold, param, output_lex_path)

# reorder lex
tgt2src_align_pair = f"{in_domain_dir}/alignment/tgttosrc/model/aligned.tgttosrc"
f_text = lex_seq
f_text_output = f"{store_data_dir}/lex.reorder"
pair_position = 0
generate_mono_data(tgt2src_align_pair, f_text, f_text_output, pair_position)

output_lex_reorder_path = f"{store_data_dir}/lex.reorder.bpe" 
apply_BPE(bpeDir, f_text_output, tgt_voc_dir, bpe_threshold, param, output_lex_reorder_path)

# split dataset
train_corpus = f"{in_domain_dir}/train.tok.tc.bpe.{src}"
dev_corpus = f"{in_domain_dir}/dev.tok.tc.bpe.{src}"
test_corpus = f"{in_domain_dir}/test.tok.tc.bpe.{src}"
train_row_num = int(return_number_of_rows_of_corpus(train_corpus))
dev_row_num = int(return_number_of_rows_of_corpus(dev_corpus))
test_row_num = int(return_number_of_rows_of_corpus(test_corpus))

for prefix in ["src", "tgt", "lex", "lex.reorder"]:
    input_corpus = f"{store_data_dir}/{prefix}.bpe"
    # train
    start_index = 0
    end_index = train_row_num
    output_corpus = f"{store_data_dir}/train.{prefix}.bpe"
    cat_specific_line(input_corpus, start_index, end_index, output_corpus)
    # dev
    start_index = train_row_num
    end_index = train_row_num + dev_row_num
    output_corpus = f"{store_data_dir}/dev.{prefix}.bpe"
    cat_specific_line(input_corpus, start_index, end_index, output_corpus)
    # test
    start_index = train_row_num + dev_row_num
    end_index = train_row_num + dev_row_num + test_row_num
    output_corpus = f"{store_data_dir}/test.{prefix}.bpe"
    cat_specific_line(input_corpus, start_index, end_index, output_corpus)

# permutation
task_list = ["aux_123_base", "aux_321_base", "aux_132_base", "aux_312_base", "aux_213_base", "aux_231_base"]

for prefix in ["train", "dev", "test"]:
    f_src = f"{store_data_dir}/{prefix}.src.bpe"
    f_lex = f"{store_data_dir}/{prefix}.lex.bpe"
    f_tgt_align = f"{store_data_dir}/{prefix}.lex.reorder.bpe"
    f_tgt = f"{store_data_dir}/{prefix}.tgt.bpe"

    # generate permutation corpus
    for task in task_list:
        output_src_path = f"{store_data_dir}/{prefix}.{task}.{src}"
        output_tgt_path = f"{store_data_dir}/{prefix}.{task}.{tgt}"
        generate_aux_seq_data(f_src, f_lex, f_tgt_align, f_tgt, task, output_src_path, output_tgt_path)

    # combine into one corpus
    for l in [src, tgt]:
        shell(f"rm -rf {store_data_dir}/{prefix}.tok.tc.bpe.{l}")
        for i, task in enumerate(task_list):
            shell(f"cat {store_data_dir}/{prefix}.{task}.{l} >> {store_data_dir}/{prefix}.tok.tc.bpe.{l}")

# generate input sequence of OOD test sets
for task in task_list: 
    input_corpus = f"{in_domain_dir}/test.tok.tc.bpe.{src}"
    output_src_path = f"{in_domain_dir}/test.tok.tc.bpe.{src}.{task}"
    generate_aux_seq_src_data(input_corpus, task, output_src_path)
    for domain in domains:
        input_corpus = f"{ood_domain_dir}/{domain}/test.tok.tc.bpe.{src}"
        output_src_path = f"{ood_domain_dir}/{domain}/test.tok.tc.bpe.{src}.{task}"
        generate_aux_seq_src_data(input_corpus, task, output_src_path)

logging.info("Done")
# ------------template 1 finished----------------------------------------- #

