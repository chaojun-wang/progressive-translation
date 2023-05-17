import os
import sys
path_to_top_dir = # path to the top directory (directory of readme)
sys.path.append(f"{path_to_top_dir}/scripts")
from data_generation import logging, shell, general_data_preprcess, build_dictionary, pre_inject_voc, train_alignment_pipe, generate_mgiza_dic_json

# public variables
mosesDir = f"{path_to_top_dir}/codes/mosesdecoder"
mgizappDir = f"{path_to_top_dir}/codes/mgiza/mgizapp/build/bin/"
bpeDir = f"{path_to_top_dir}/codes/subword-nmt/subword_nmt"
nematus_dir = f"{path_to_top_dir}/codes/nematus"

data_dir = f"{path_to_top_dir}/data"
data_dir_iwslt = f"{data_dir}/iwslt"
data_dir_domain = f"{data_dir}/iwslt/domain_test_data"
domains = ["medical", "it", "law"]
bpe_operations = 10000
bpe_threshold = 1
min_length = 5
max_length = 100

lang1 = "de"
lang2 = "en"

id_bpe_model_dir = "/misc/projdata1/info_fil/cwang/platform/data/emnlp2022/iwslt/iwslt-en-de/model"

# Start
# Start logging.
logging.info("Start")
logging.info(f"pid is: {os.getpid()}")

# create directory
shell(f"mkdir -p {data_dir}")
shell(f"mkdir -p {data_dir_iwslt}")
shell(f"mkdir -p {data_dir_domain}")

# download and preprocess IWSLT dataset
shell(f"wget -O {data_dir_iwslt}/emnlp2021-data.tar.gz http://www.dlsi.ua.es/~vmsanchez/emnlp2021-data.tar.gz")
shell(f"tar xvzf {data_dir_iwslt}/emnlp2021-data.tar.gz -C {data_dir_iwslt}")
shell(f"mv {data_dir_iwslt}/data/* {data_dir_iwslt}")
shell(f"rm -rf {data_dir_iwslt}/data")
shell(f"mv {data_dir_iwslt}/iwslt-en-de/IWSLT14.TED.tst2013.en-de.de {data_dir_iwslt}/iwslt-en-de/dev.de")
shell(f"mv {data_dir_iwslt}/iwslt-en-de/IWSLT14.TED.tst2013.en-de.en {data_dir_iwslt}/iwslt-en-de/dev.en")
shell(f"mv {data_dir_iwslt}/iwslt-en-de/IWSLT14.TED.tst2014.en-de.en {data_dir_iwslt}/iwslt-en-de/test.en")
shell(f"mv {data_dir_iwslt}/iwslt-en-de/IWSLT15.TED.tst2014.en-de.de {data_dir_iwslt}/iwslt-en-de/test.de")

dataset_indicator = {"train": True, "dev": True, "test": True}
clean_corpus_param = {"min_length": min_length, "max_length": max_length, "ratio": None}
bpe_param = {"bpe_operations": bpe_operations, "bpe_threshold": bpe_threshold}
general_data_preprcess(f"{data_dir_iwslt}/iwslt-en-de", mosesDir, bpeDir, lang1, lang2, clean_corpus_param, bpe_param, dataset_indicator)

# build dictionary
source_corpus = f"{data_dir_iwslt}/iwslt-en-de/train.tok.tc.bpe.{lang1}"
target_corpus = f"{data_dir_iwslt}/iwslt-en-de/train.tok.tc.bpe.{lang2}"
tie_output = f"{data_dir_iwslt}/iwslt-en-de/train.tok.tc.bpe.both"
build_dictionary(nematus_dir, source_corpus, target_corpus, True, tie_output)
dic_path = f"{data_dir_iwslt}/iwslt-en-de/train.tok.tc.bpe.both.json"
output_path = f"{data_dir_iwslt}/iwslt-en-de/train.tok.tc.bpe.both.special_tokens.json"
pre_inject_voc(dic_path, output_path)

# generate alignment
align_types = ["tgttosrc", "srctotgt", "intersection"]
alignment_dir = f"{data_dir_iwslt}/iwslt-en-de/alignment"
shell(f"mkdir -p {alignment_dir}")
for l in [lang1, lang2]:
    shell(f"cat {data_dir_iwslt}/iwslt-en-de/train.tok.tc.{l} {data_dir_iwslt}/iwslt-en-de/dev.tok.tc.{l} {data_dir_iwslt}/iwslt-en-de/test.tok.tc.{l} > {alignment_dir}/total.{l}")
input_path_prefix = f"{alignment_dir}/total"
for align_type in align_types:
    align_output_path = f"{alignment_dir}/{align_type}"
    train_alignment_pipe(align_type, input_path_prefix, align_output_path, lang1, lang2, mosesDir, mgizappDir)
generate_mgiza_dic_json(f"{data_dir_iwslt}/iwslt-en-de/alignment/intersection/model/lex.e2f", f"{data_dir_iwslt}/iwslt-en-de/alignment/intersection/model/lex.e2f.json")


# download domain test set
shell(f"wget -N https://files.ifi.uzh.ch/cl/archiv/2019/clcontra/opus_robustness_data_v2.tar.xz --no-check-certificate -P {data_dir_domain}")
shell(f"tar -xvf {data_dir_domain}/opus_robustness_data_v2.tar.xz -C {data_dir_domain}")
shell(f"mv {data_dir_domain}/opus_robustness_data/* {data_dir_domain}/")
shell(f"rm -rf {data_dir_domain}/opus_robustness_data")

dataset_indicator = {"train": False, "dev": False, "test": True}
clean_corpus_param = {"min_length": None, "max_length": None, "ratio": None}
bpe_param = {"bpe_operations": bpe_operations, "bpe_threshold": bpe_threshold}
trained_data_dir = f"{data_dir_iwslt}/iwslt-en-de/"
for domain in domains:
    dataset_dir = f"{data_dir_domain}/{domain}"
    general_data_preprcess(dataset_dir, mosesDir, bpeDir, lang1, lang2, clean_corpus_param, bpe_param, dataset_indicator, trained_data_dir)

logging.info("Done")
