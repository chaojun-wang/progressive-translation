import os
import json
import copy
import random
import logging
import itertools
from numpy.random import choice
from typing import OrderedDict

level = logging.INFO
logging.basicConfig(level=level, format='%(levelname)s: [%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def shell(command):
    with os.popen(command) as f:
        ret = f.read()
    if ret:
        logging.info(ret)

def corpus_read_lines(corpus_path):
    with open(corpus_path) as f:
        output = f.readlines()
    return output

def tokenize(mosesDir, input_corpus, language, output_corpus):
    logging.info(f"tokenize {input_corpus}")
    shell(f"cat {input_corpus} | \
            {mosesDir}/scripts/tokenizer/normalize-punctuation.perl -l {language} | \
            {mosesDir}/scripts/tokenizer/tokenizer.perl -a -l {language} > {output_corpus}")

def clean_corpus(mosesDir, input_corpus_prefix, src, tgt, output_corpus_prefix, min_length, max_length, ratio):
    logging.info(f"clean {input_corpus_prefix} with maxlen: {max_length}, minlen: {min_length} and ratio: {ratio}")
    if ratio is not None:
        shell(f"{mosesDir}/scripts/training/clean-corpus-n.perl -ratio {ratio} {input_corpus_prefix} {src} {tgt} {output_corpus_prefix} {min_length} {max_length}")
    else:
        shell(f"{mosesDir}/scripts/training/clean-corpus-n.perl {input_corpus_prefix} {src} {tgt} {output_corpus_prefix} {min_length} {max_length}")

def move_out_useless_data(data_dir, lang1, lang2):
    shell(f"mkdir -p {data_dir}/useless")
    for l in [lang1, lang2]:
        shell(f"mv {data_dir}/train.{l} {data_dir}/useless/train.{l}")
        shell(f"mv {data_dir}/train.tok.{l} {data_dir}/useless/train.tok.{l}")

def train_truecase(mosesDir, input_corpus, parameter_storage_path):
    logging.info(f"train truecase for {input_corpus}")
    shell(f"{mosesDir}/scripts/recaser/train-truecaser.perl -corpus {input_corpus} -model {parameter_storage_path}")

def apply_truecase(mosesDir, input_corpus, parameter_storage_path, output_corpus):
    logging.info(f"apply truecase for {input_corpus} with parameters stored at {parameter_storage_path}")
    shell(f"{mosesDir}/scripts/recaser/truecase.perl -model {parameter_storage_path} < {input_corpus} > {output_corpus}")

def train_BPE(bpeDir, input_src_corpus, input_tgt_corpus, src_voc_stor_path, tgt_voc_stor_path, bpe_operations, output_param):
    logging.info(f"train BPE for {input_src_corpus} and {input_tgt_corpus}, with bpe_operations: {bpe_operations}")
    shell(f"{bpeDir}/learn_joint_bpe_and_vocab.py -i {input_src_corpus} {input_tgt_corpus} --write-vocabulary {src_voc_stor_path} {tgt_voc_stor_path} -s {bpe_operations} -o {output_param}")

def apply_BPE(bpeDir, input_corpus, vocabulary_path, vocabulary_threshold, param_path, output_corpus):
    logging.info(f"apply BPE for {input_corpus}, with vocabulary threshold: {vocabulary_threshold} and parameters stored at {param_path}")
    shell(f"{bpeDir}/apply_bpe.py -c {param_path} --vocabulary {vocabulary_path} --vocabulary-threshold {vocabulary_threshold} < {input_corpus} > {output_corpus}")

def general_data_preprcess(dataset_dir, mosesDir, bpeDir, lang1, lang2, clean_corpus_param, bpe_param, dataset_indicator, trained_data_dir=None):
    """
    lang1 -> src
    lang2 —> tgt
    (1) enable preprocess without train truecase and bpe (by setting trained_data_dir as the path of dataset which has trained parameters)
    (2) enable select the type of dataset to be preprocessed (e.g. only preprocess dev, test by setting dataset_indicator dictionary)
    """
    param_dir = trained_data_dir if trained_data_dir is not None else dataset_dir
    min_length, max_length, ratio = clean_corpus_param["min_length"], clean_corpus_param["max_length"], clean_corpus_param["ratio"]
    bpe_operations, bpe_threshold, = bpe_param["bpe_operations"], bpe_param["bpe_threshold"]
    # tokenize
    for prefix in ["train", "dev", "test"]:
        if dataset_indicator[prefix] == False:
            continue
        for l in [lang1, lang2]:
            input_corpus = f"{dataset_dir}/{prefix}.{l}"
            output_corpus = f"{dataset_dir}/{prefix}.tok.{l}"
            tokenize(mosesDir, input_corpus, l, output_corpus)
    # clean corpus
    if dataset_indicator["train"] != False:
        input_corpus_prefix = f"{dataset_dir}/train.tok"
        output_corpus_prefix = f"{dataset_dir}/train.tok.cleaned"
        clean_corpus(mosesDir, input_corpus_prefix, lang1, lang2, output_corpus_prefix, min_length, max_length, ratio=ratio)
    # arrange directory
    if dataset_indicator["train"] != False:
        directory_path = f"{dataset_dir}/"
        move_out_useless_data(directory_path, lang1, lang2)
        for l in [lang1, lang2]:
            original_name = f"{dataset_dir}/train.tok.cleaned.{l}"
            target_name = f"{dataset_dir}/train.tok.{l}"
            shell(f"mv {original_name} {target_name}")
    # train truecase
    if dataset_indicator["train"] != False and trained_data_dir == None:
        shell(f"mkdir -p {dataset_dir}/model")
        for l in [lang1, lang2]:
            input_path = f"{dataset_dir}/train.tok.{l}"
            parameter_path = f"{dataset_dir}/model/truecase-model.{l}"
            train_truecase(mosesDir, input_path, parameter_path)
    # apply truecase
    for prefix in ["train", "dev", "test"]:
        if dataset_indicator[prefix] == False:
            continue
        for l in [lang1, lang2]:
            input_path = f"{dataset_dir}/{prefix}.tok.{l}"
            output_path = f"{dataset_dir}/{prefix}.tok.tc.{l}"
            parameter_path = f"{param_dir}/model/truecase-model.{l}"
            apply_truecase(mosesDir, input_path, parameter_path, output_path)
    # train BPE
    if dataset_indicator["train"] != False and trained_data_dir == None:
        input_corpus_src = f"{dataset_dir}/train.tok.tc.{lang1}"
        input_corpus_tgt = f"{dataset_dir}/train.tok.tc.{lang2}"
        src_voc_storage = f"{dataset_dir}/model/vocab.{lang1}"
        tgt_voc_storage = f"{dataset_dir}/model/vocab.{lang2}"
        output_param_path = f"{dataset_dir}/model/{lang1}{lang2}.bpe"
        train_BPE(bpeDir, input_corpus_src, input_corpus_tgt, src_voc_storage, tgt_voc_storage, bpe_operations, output_param_path)
    # apply BPE
    for prefix in ["train", "dev", "test"]:
        if dataset_indicator[prefix] == False:
            continue
        for l in [lang1, lang2]:
            input_path = f"{dataset_dir}/{prefix}.tok.tc.{l}"
            output_path = f"{dataset_dir}/{prefix}.tok.tc.bpe.{l}"
            parameter_path = f"{param_dir}/model/{lang1}{lang2}.bpe"
            vocabulary_path = f"{param_dir}/model/vocab.{l}"
            apply_BPE(bpeDir, input_path, vocabulary_path, bpe_threshold, parameter_path, output_path)

def build_dictionary(nematus_dir, source_corpus, target_corpus, tie, tie_output=None):
    logging.info(f"build dictionary for source corpus: {source_corpus} and target corpus {target_corpus}, tie src tgt or not: {tie}")
    if tie:
        assert tie_output is not None
        shell(f"rm -rf {tie_output}")
        shell(f"cat {source_corpus} {target_corpus} > {tie_output}")
        shell(f"{nematus_dir}/data/build_dictionary.py {tie_output}")
    else:
        shell(f"{nematus_dir}/data/build_dictionary.py {source_corpus}")
        shell(f"{nematus_dir}/data/build_dictionary.py {target_corpus}")

def convert_subword_in_sublist(src_list, bpe_separator):
    new_line = []
    in_subword = False
    for subword in src_list:
        if subword[-2:] == bpe_separator:
            if not in_subword:
                tmp_list = [subword]
                in_subword = True
            else:
                tmp_list.append(subword)
        else:
            if not in_subword:
                new_line.append([subword])
            else:
                in_subword = False
                tmp_list.append(subword)
                new_line.append(tmp_list)
    return new_line

def convert_sublist_back_to_list(src_list):
    new_line = []
    for sublist in src_list:
        new_line.extend(sublist)
    return new_line

def return_number_of_rows_of_corpus(corpus_path):
    with open(corpus_path) as f:
        corpus = f.readlines()
    return len(corpus)

def cat_specific_line(corpus, start_index, end_index, output_corpus):
    with open(corpus) as f:
        file = f.readlines()
    output = "".join(file[start_index: end_index])
    with open(output_corpus, "w") as f:
        f.write(output)

def generate_align(mosesDir, mgizappDir, corpus_path_prefix, src_l, tgt_l, align_output_path, align_type):
    """
    train-model.perl: -e is the target language, -f is the source language.
    --alignment represents the type of alignment, tgttosrc, srctotgt, intersection
    in all types, the output is in the format of <pos_of_src>-<pos_of_tgt> for all,
    and for xxtoxx, the previous one is the base so that it's ordered (possibly jump over position) and the previous one's sequence is not repeated; for intersection, the order is based on source and both source and target does not have
    repeat position.
    """
    logging.info(f"generate alignment pair sequence for {corpus_path_prefix}, with alignment type: {align_type}")
    njobs = 32
    shell(f"{mosesDir}/scripts/training/train-model.perl  --alignment {align_type} --root-dir {align_output_path} --corpus {corpus_path_prefix} -e {tgt_l} -f {src_l} --mgiza --mgiza-cpus={njobs} --parallel --first-step 1 --last-step 4 --external-bin-dir {mgizappDir} --sort-compress gzip")

def train_alignment_pipe(align_type, input_path_prefix, align_output_path, lang1, lang2, mosesDir, mgizappDir):
    # align_type_list e.g. "tgttosrc"
    # input_path_prefix e.g. f"{data_dir_iwslt}/iwslt-en-de/train.tok.tc.bpe"
    # align_output_path e.g. f"{data_dir_iwslt}/iwslt-en-de/alignment/bpenized/tgttosrc"
    shell(f"mkdir -p {align_output_path}")
    generate_align(mosesDir, mgizappDir, input_path_prefix, lang1, lang2, align_output_path, align_type)

def generate_mgiza_dic_json(dict_f, f_output, one2one=False, no_distribution=True):
    src_tgt_map = OrderedDict()
    with open(dict_f) as f:
        dict = f.readlines()
    for line in dict:
        line = line.strip("\n").split()
        src = line[0]
        tgt = line[1]
        pro = line[2]
        if tgt != "NULL" and src != "NULL":
            if src not in src_tgt_map:
                src_tgt_map[src] = []
            src_tgt_map[src].append((tgt, pro))
    if one2one and no_distribution:
        src_tgt_map_new = {}
        for src_tok, tgt_list in src_tgt_map.items():
            src_tgt_map_new[src_tok]=[max(tgt_list, key=lambda x:x[1])[0]]
        src_tgt_map = src_tgt_map_new
    elif no_distribution:
        src_tgt_map_new = {}
        for src_tok, tgt_list in src_tgt_map.items():
            tgt_list = sorted(tgt_list, key=lambda x:x[1], reverse=True)
            src_tgt_map_new[src_tok]=[i[0] for i in tgt_list]
        src_tgt_map = src_tgt_map_new
    with open(f_output, 'w') as f:
        json.dump(src_tgt_map, f, indent=2, ensure_ascii=False)

def sample_a_lex_token(dic, src_token, distribution, lower_src, fix_lexical_map=False, aligned_token=None, return_copy_src_or_not=False):
    copy_src = False
    if dic == None:
        return src_token
    if lower_src and src_token not in dic and src_token.lower() in dic:
        src_token = src_token.lower()
    if src_token in dic:
        copy_src = False
        if distribution:
            value_content = dic[src_token]
            list_of_candidates = [i[0] for i in value_content]
            probability_distribution = [float(i[1]) for i in value_content]
            probability_distribution = [float(i)/sum(probability_distribution) for i in probability_distribution]
            draw = choice(list_of_candidates, 1, p=probability_distribution)[0]
        elif aligned_token:
            if len(dic[src_token]) >= 1 and aligned_token in dic[src_token]:
                draw = aligned_token
            else:
                draw = dic[src_token][0]
        else:
            if fix_lexical_map:
                draw = dic[src_token][0]
            else:
                draw = random.choice(dic[src_token])
    else:
        copy_src = True
        draw = src_token
    if return_copy_src_or_not:
        return draw, copy_src
    else:
        return draw

def sample_a_lex_seq(dic, source_seq, distribution=False, lower_src=False, fix_lexical_map=False, ignore_ratio=0, aligned_token_seq=None, return_copy_src_or_not=False):
    lexical_map_sequence = []
    ignore_index = [i for i in range(len(source_seq))]
    ignore_index = set(random.sample(ignore_index, round(len(source_seq) * ignore_ratio)))

    for i, src_token in enumerate(source_seq):
        if i in ignore_index:
            lexical_map_sequence.append(src_token)
        else:
            if aligned_token_seq:
                aligned_token = aligned_token_seq[i]
            else:
                aligned_token = None
            if return_copy_src_or_not:
                draw, copy_src = sample_a_lex_token(dic, src_token, distribution, lower_src, fix_lexical_map, aligned_token, return_copy_src_or_not)
                lexical_map_sequence.append((draw, copy_src))
            else:
                draw = sample_a_lex_token(dic, src_token, distribution, lower_src, fix_lexical_map, aligned_token)
                lexical_map_sequence.append(draw)
    return lexical_map_sequence

def generate_aligned_token_seq(f_src, f_align, f_tgt):
    src = corpus_read_lines(f_src)
    ali = corpus_read_lines(f_align)
    tgt = corpus_read_lines(f_tgt)
    aligned_token_seq_corpus = []
    for i in range(len(src)):
        src_line = src[i].strip("\n").split()
        aligned_token_seq = [None for _ in range(len(src_line))]
        ali_line = ali[i].strip("\n").split()
        tgt_line = tgt[i].strip("\n").split()
        for src2tgt_pair in ali_line:
            src_index = int(src2tgt_pair.split("-")[0])
            tgt_index = int(src2tgt_pair.split("-")[1])
            aligned_token_seq[src_index] = tgt_line[tgt_index]
        aligned_token_seq_corpus.append(aligned_token_seq)
    assert len(aligned_token_seq_corpus) == len(src) == len(ali) == len(tgt)
    return aligned_token_seq_corpus

def generate_lexical_map_seq_dicbased(f_src, dic_file, output_file, lower_src=False, distribution=False, fix_lexical_map=False, ignore_ratio=0, f_align=False, f_tgt=False, bpenized=False):
    src = open(f_src)
    with open(dic_file, 'r') as f:
        dic = json.load(f)
    if f_align:
        assert f_tgt is not False
        aligned_token_seq_corpus = generate_aligned_token_seq(f_src, f_align, f_tgt)
    else:
        aligned_token_seq_corpus = None
    lexical_map_seq_file_string = ""
    for i, text_line in enumerate(src):
        source_seq = text_line.strip("\n").split()
        if bpenized:
            formatted_source_seq = []
            for sublist in convert_subword_in_sublist(source_seq, "@@"):
                formatted_source_seq.append(" ".join(sublist))
            source_seq = formatted_source_seq

        if aligned_token_seq_corpus:
            aligned_token_seq = aligned_token_seq_corpus[i]
        else:
            aligned_token_seq = None
        lexical_map_sequence = sample_a_lex_seq(dic, source_seq, distribution=distribution, lower_src=lower_src, fix_lexical_map=fix_lexical_map, ignore_ratio=ignore_ratio, aligned_token_seq=aligned_token_seq)
        lexical_map_seq_file_string += " ".join(lexical_map_sequence) + "\n"
    with open(output_file, "w") as f:
        f.write(lexical_map_seq_file_string)

def generate_aux_seq_data(src, tgt_lex, tgt_align, tgt, task, output_src_path, output_tgt_path, reverse_subseq=False):
    src_file = open(src).readlines()
    tgt_lex_file = open(tgt_lex).readlines()
    tgt_align_file = open(tgt_align).readlines()
    tgt_file = open(tgt).readlines()
    assert len(src_file) == len(tgt_lex_file) == len(tgt_align_file) == len(tgt_file)

    num_file_map = {1: tgt_lex_file, 2: tgt_align_file, 3: tgt_file}
    num_code_map = {1: "<LEX>", 2: "<ALI>", 3: "<TGT>"}
    seq_order = list(task.split("_")[1])
    seq_property = task.split("_")[2].upper()

    output_src = ""
    output_tgt = ""
    for i in range(len(src_file)):
        src_seq = src_file[i]
        output_src += "<{}>".format("".join(seq_order)) + " " + "<{}>".format(seq_property) + " " + src_seq

        first_seq = num_file_map[int(seq_order[0])][i].strip("\n")
        second_seq = num_file_map[int(seq_order[1])][i].strip("\n")
        third_seq = num_file_map[int(seq_order[2])][i].strip("\n")
        if reverse_subseq:
            first_seq = " ".join(reversed(first_seq.split()))
            second_seq = " ".join(reversed(second_seq.split()))
            third_seq = " ".join(reversed(third_seq.split()))
        output_tgt += num_code_map[int(seq_order[0])] + " " + first_seq + " " + num_code_map[int(seq_order[1])] + " " + second_seq + " " + num_code_map[int(seq_order[2])] + " " + third_seq + "\n"

    with open(output_src_path, "w") as f:
        f.write(output_src)
    with open(output_tgt_path, "w") as f:
        f.write(output_tgt)

def generate_aux_seq_target_data_noalign(src, tgt_lex, tgt, task, output_src_path, output_tgt_path):
    src_file = open(src).readlines()
    tgt_lex_file = open(tgt_lex).readlines()
    tgt_file = open(tgt).readlines()
    assert len(src_file) == len(tgt_lex_file) == len(tgt_file)

    num_file_map = {1: tgt_lex_file, 3: tgt_file}
    num_code_map = {1: "<LEX>", 3: "<TGT>"}
    seq_order = list(task.split("_")[1])
    seq_property = task.split("_")[2].upper()

    output_src = ""
    output_tgt = ""
    for i in range(len(src_file)):
        src_seq = src_file[i]
        output_src += "<{}>".format("".join(seq_order)) + " " + "<{}>".format(seq_property) + " " + src_seq

        first_seq = num_file_map[int(seq_order[0])][i].strip("\n")
        second_seq = num_file_map[int(seq_order[1])][i].strip("\n")
        output_tgt += num_code_map[int(seq_order[0])] + " " + first_seq + " " + num_code_map[int(seq_order[1])] + " " + second_seq + "\n"
    
    with open(output_src_path, "w") as f:
        f.write(output_src)
    with open(output_tgt_path, "w") as f:
        f.write(output_tgt)

def generate_aux_seq_src_data(src, task, output_src_path):
    src_file = open(src).readlines()
    seq_order = list(task.split("_")[1])
    seq_property = task.split("_")[2].upper()

    output_src = ""
    for i in range(len(src_file)):
        src_seq = src_file[i]
        output_src += "<{}>".format("".join(seq_order)) + " " + "<{}>".format(seq_property) + " " + src_seq

    with open(output_src_path, "w") as f:
        f.write(output_src)

def pre_inject_voc(dic_path, output_path):
    control_tokens = ["<BASE>", "<MONO>", "<REVERSE>", "<REPLACE>", "<LEX>", "<ALI>", "<TGT>"]
    order_tokens = ["<{}{}{}>".format(i,j,k) for i,j,k in list(itertools.permutations([1,2,3],3))]
    spcecial_tokens = control_tokens + order_tokens

    with open(dic_path, 'r') as f:
        dic = json.load(f)

    voc_num = len(dic)
    for token in spcecial_tokens:
        if token not in dic:
            dic[token] = voc_num
            voc_num += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dic, f, indent=2, ensure_ascii=False)

def mono_seq(align_pair_seq, text_seq, pair_position):
    # make a pair of parallel sequences monotonic according to their word-alignment sequence
    # by reorderring one of the sequence in the pair
    # Note:
    #   pair_position is 1 if text_seq is target, align_pair_seq is then from src2tgt;
    #                    0 if text_seq is source, align_pair_seq is then from tgt2src
    output_test = copy.deepcopy(text_seq)
    seen_text_seq_index = set()
    seen_max_text_seq_id = 0
    old_id2new_id = {}
    for idx in range(len(text_seq)):
        old_id2new_id[idx] = idx
    for ali_pair in align_pair_seq:
        text_seq_id = int(ali_pair.split("-")[pair_position])
        if text_seq_id in seen_text_seq_index:
            continue
        seen_max_text_seq_id = max(seen_max_text_seq_id, text_seq_id)
        seen_text_seq_index.add(text_seq_id)
        if text_seq_id < seen_max_text_seq_id:
            output_test.pop(old_id2new_id[text_seq_id])
            output_test.insert(seen_max_text_seq_id, text_seq[text_seq_id])
            # update map
            start = text_seq_id+1
            for i in range(start, seen_max_text_seq_id+1):
                old_id2new_id[i] -= 1
    return output_test

def generate_mono_data(srctotgt_align_pair, f_tgt, f_tgt_output, pair_position=1):
    logging.info(f"generate mono mlt data for {f_tgt}")
    with open(srctotgt_align_pair) as f:
        align_pair = f.readlines()
    with open(f_tgt) as f:
        tgt_corpus = f.readlines()
    assert len(align_pair) == len(tgt_corpus)

    output_tgt = ""
    for index in range(len(align_pair)):
        align_pair_line = align_pair[index].strip("\n").split()
        tgt_line = tgt_corpus[index].strip("\n").split()
    
        new_tgt_line = mono_seq(align_pair_line, tgt_line, pair_position)
        output_tgt += " ".join(new_tgt_line) + "\n"

    with open(f_tgt_output, "w") as f:
        f.write(output_tgt)
