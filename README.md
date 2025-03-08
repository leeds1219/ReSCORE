<div align="center">
  <img src="assets/korea_university.svg" alt="Korea University Logo">
  <img src="assets/miil.svg" alt="MIIL Logo">
  <img src="assets/naver_ai_lab.svg" alt="Naver AI Logo">
  <img src="assets/naver_cloud.svg" alt="Naver Cloud Logo">
  <img src="assets/richmond_university.svg" alt="Richmond Logo">
</div>

# ReSCORE: Label-free Iterative Retriever Training for Multi-hop Question Answering with Relevance-Consistency Supervision

[[arXiv](https://arxiv.org/abs/)] [[Project](https://leeds1219.github.io/ReSCORE/)] <br>

by [Dosung Lee](https://leeds1219.github.io/)\*, [Wonjun Oh](https://github.com/owj0421)\*, [Boyoung Kim](bykimby.github.io), [Minyoung Kim](https://github.com/EuroMinyoung186), [Joonsuk Park](http://www.mathcs.richmond.edu/~jpark/)†, [Paul Hongsuck Seo](https://miil.korea.ac.kr/)†

This is our official implementation of ReSCORE: Label-free Iterative Retriever Training for Multi-hop Question Answering with Relevance-Consistency Supervision! 

## Introduction
![Figure](assets/figure.png)
Multi-hop question answering (MHQA) requires reasoning across multiple documents, making dense retriever training challenging due to query variability. We propose ReSCORE, a method that trains dense retrievers without labeled data by leveraging LLMs to assess document relevance and consistency with answers.

For further details, please check out our [Paper](https://arxiv.org/abs/) and our [Project](https://leeds1219.github.io/ReSCORE/) page.

## :fire:TODO
- [x] Clean Code
- [ ] Check Typo
- [ ] Build Project Page
- [ ] Upload Revised Paper to arXiv, add Citation & Acknowledgement

## Installation
```
pip install -r requirements.txt
```

You need permission to access the [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) model, or you can modify the [Script](/source/module/generate/llama.py) to use your own LLM.

We conducted all experiments using Python 3.10.12 on an NVIDIA A100 HBM2 40GB PCIe GPU, and the environments are listed in [my_packages.txt](./my_packages.txt), so please refer to [my_packages.txt](./my_packages.txt) if any issues arise.

## Data Preparation
```bash
# Download MHQA datasets
sh script/download/multihop_raw_data.sh

# Preprocess and build Retrieval DB
sh script/download/build.sh
```

## Training
```
# Training
python -m source.run.train
--running_name {train}
--dataset {dataset}
```

<!--<img src="assets/loss.png" width="50%" />
We selected the checkpoints corresponding to the lowest validation loss within a single epoch.-->

#### Model Weights
| Model Weights | Link |
|--------------|------|
| Contriever-MSMARCO | [🔗 Click here](https://huggingface.co/facebook/contriever-msmarco) |
| IQATR-Musique | [🔗 Click here](https://huggingface.co/Lee1219/iqatr-musique) |
| IQATR-HotpotQA | [🔗 Click here](https://huggingface.co/Lee1219/iqatr-hotpotqa) |
| IQATR-2WikiMultiHopQA | [🔗 Click here](https://huggingface.co/Lee1219/iqatr-2wikimhqa) |

## Inference
```
# Inference
python -m source.run.inference
--method {base_or_iqatr}
--running_name {inference}
--dataset {dataset}
```

## Acknowledgement
This research was supported by ...

This project includes code from [Contriever](https://github.com/facebookresearch/contriever), [DPR](https://github.com/facebookresearch/DPR), and [IRCoT](https://github.com/StonyBrookNLP/ircot).

## Citation
```BibTeX

```
