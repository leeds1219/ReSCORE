# ReSCORE: Label-free Iterative Retriever Training for Multi-hop Question Answering with Relevance-Consistency Supervision

<div style="display: flex; justify-content: space-between; align-items: center;">
  <img src="assets/ku-logo.png" alt="korea" height="30">
  <img src="assets/miil.png" alt="miil" height="30">
</div>

[[arXiv](https://leeds1219.github.io/)] [[Project](https://leeds1219.github.io/)] <br>

by [Dosung Lee](https://leeds1219.github.io/)\*, [Wonjun Oh](https://github.com/owj0421)\*, [Boyoung Kim](bykimby.github.io), [Minyoung Kim](https://github.com/EuroMinyoung186), [Joonsuk Park](http://www.mathcs.richmond.edu/~jpark/)†, [Paul Hongsuck Seo](https://miil.korea.ac.kr/)†

This is our official implementation of ReSCORE! 

## Introduction
![Figure](assets/figure.png)

## :fire:TODO
- [x] 
- [ ] 
- [ ] 
- [ ] 
- [ ] 
- [ ]

## Installation
```pip install -r requirements.txt```

You need [Llama-3.1-8B-Instruct](https://huggingface.co/TheBloke/Llama-3.1-8B-Instruct) model permission or you can modify the code a bit for your own LLM.

## Data Preparation
```bash
# Download MHQA datasets
sh script/download/multihop_raw_data.sh

# Preprocess and build Retrieval DB
sh script/download/build.sh
```

## Training
```python -m source.run.train --dataset ...```

### Model Weights
| Model Weights | Link |
|--------------|------|
| IQATR-Musique | [🔗 Click here](https://huggingface.co/Lee1219/iqatr-musique) |
| IQATR-HotpotQA | [🔗 Click here](https://huggingface.co/Lee1219/iqatr-hotpotqa) |
| IQATR-2WikiMultiHopQA | [🔗 Click here](https://huggingface.co/Lee1219/iqatr-2wikimhqa) |

## Inference
```python -m source.run.inference --dataset ...```

## Acknowledgement

## Citing

```BibTeX

```
