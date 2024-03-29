# Progressive Translation: Improving Domain Robustness of Neural Machine Translation with Intermediate Sequences

Training scripts of [Progressive Translation: Improving Domain Robustness of Neural Machine Translation with Intermediate Sequences](https://aclanthology.org/2023.findings-acl.601/).


## 1. Requirements
python 3.9

tensorflow 2.4

sacrebleu 2.1

gpuinfo

## 2. Download Software and Create Directory
```
bash scripts/setup_software.sh
```

## 3. Download and Preprocess Datasets (for IWSLT'14)
```
python scripts/iwslt/data.py
```

## 4. Generate Training and Testing data for Progressive Translation
```
python scripts/iwslt/pt/data.py
```

## 5. Train and Evaluate the NMT model
```
# Baseline:
bash scripts/iwslt/baseline/train.sh

# Progressive Translation (full):
bash scripts/iwslt/pt/train.sh
```

