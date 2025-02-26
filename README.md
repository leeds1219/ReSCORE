<div style="display: flex; justify-content: center; align-items: center; gap: 50px;">
  <img src="assets/ku-logo.png" alt="korea" height="30">
  <img src="assets/miil.png" alt="miil" height="30">
</div>

# ReSCORE: Label-free Iterative Retriever Training for Multi-hop Question Answering with Relevance-Consistency Supervision

[[arXiv]()] [[Project]()] <br>

by [Dosung Lee](https://leeds1219.github.io/about/)\*, [Wonjun Oh](https://github.com/owj0421)\*, [Boyoung Kim](bykimby.github.io), [Minyoung Kim](https://github.com/EuroMinyoung186), [Joonsuk Park](http://www.mathcs.richmond.edu/~jpark/)†, [Paul Hongsuck Seo](https://miil.korea.ac.kr/)†

This is our official implementation of ReSCORE! 

## Introduction
![Figure](assets/figure.png)
Multi-hop question answering (MHQA) requires reasoning across multiple documents, making dense retriever training challenging due to query variability. We propose ReSCORE, a method that trains dense retrievers without labeled data by leveraging LLMs to assess document relevance and consistency with answers.

For further details, please check out our [Paper]() and our [Project]() page.

## :fire:TODO
- [x] Clean Code
- [ ] Check Typo
- [ ] Build Project Page
- [ ] Upload Paper to arXiv, add Citation & Acknowledgement

## Installation
```
pip install -r requirements.txt
```

You need permission to access the [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) model, or you can modify the [Script](/source/module/generate/llama.py) to use your own LLM.

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
--generation_model_name {generator_model}
--generation_max_batch_size {depends_on_gpu}
--retrieval_count {num_docs_to_retrieve}
--retrieval_query_type {option_to_append_thoughts}
--retrieval_query_model_name_or_path {retriever_model}
--max_num_thought {max_iteration}
--batch_size {depends_on_gpu}
--lr {learning_rate}
--temperature_r {retriever_norm_temperature}
--temperature_lm {generator_norm_temperature}
```

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
--running_name {inference}
--dataset {dataset}
--generation_model_name {generator_model}
--generation_max_batch_size {depends_on_gpu}
--retrieval_count {num_docs_to_retrieve}
--retrieval_query_type {option_to_append_thoughts, full or last only}
--retrieval_query_model_name_or_path {retriever_model}
--max_num_thought {max_iteration}
```

## Acknowledgement
This research was supported by ...

This project includes code from [Contriever](https://github.com/facebookresearch/contriever), [DPR](https://github.com/facebookresearch/DPR), and [IRCoT](https://github.com/StonyBrookNLP/ircot).

## Citation
```BibTeX

```
