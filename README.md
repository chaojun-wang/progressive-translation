# Progressive Translation: Improving Domain Robustness of Neural Machine Translation with Intermediate Sequences

Training scripts of [Progressive Translation: Improving Domain Robustness of Neural Machine Translation with Intermediate Sequences](https://arxiv.org/abs/2305.09154).


## 1. Requirements
python 3.9

tensorflow 2.4

sacrebleu 2.1

gpuinfo

## 2. Download Software
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

