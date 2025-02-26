import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import copy
import warnings
import argparse
from argparse import ArgumentParser
from typing import List
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader
from source.utility.data_utils import (
    clean_and_create_dir, 
    load_data_from_jsonl
)
from source.pipeline.config import PipelineConfig
from source.pipeline.controller import PipelineController
from source.pipeline.state import QuestionState

from source.pipeline.step.retrieval import (
    RetrievalStep
)
from source.pipeline.step.training import (
    TrainStep
)
from source.pipeline.step.generation import (
    GenerationStep, 
    AnswerGenerateOutputParser, 
    AnswerGeneratePromptGenerator,
    ThoughtGenerateOutputParser,
    ThoughtGeneratePromptGenerator,
)
from source.pipeline.step.end import (
    EndStep,
)
from source.module.generate.llama import (
    LlamaGenerator,
    LlamaGeneratorConfig
)
from source.module.retrieve.dense import (
    DenseRetriever,
    DenseRetrieverConfig
)
from source.module.index.index import (
    Indexer,
    IndexerConfig
)
from source.evaluation.evaluate import (
    evaluate_by_dicts,
    official_evaluate_by_dicts,
)

def collate_question_states(batch: List[QuestionState]) -> List[QuestionState]:
    return batch  

class QuestionStateDataset(Dataset):
    def __init__(self, start_states):
        self.start_states = start_states

    def __len__(self):
        return len(self.start_states)

    def __getitem__(self, idx):
        return self.start_states[idx]

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline Configuration")

    # General Config
    parser.add_argument('--method', type=str, default="rescore", help='rescore')
    parser.add_argument('--running_name', type=str, help='Name for the running experiment')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
    parser.add_argument('--seed', type=int, default=100, help='Random seed')
    parser.add_argument('--dataset', choices=['hotpotqa', '2wikimultihopqa', 'musique'], default='musique', help='Dataset')
    parser.add_argument('--dataset_split', choices=['train', 'dev', 'test'], default='train', help='Dataset split')
    parser.add_argument('--pipeline_type', choices=['single_retrieval', 'multi_retrieval', 'no_retrieval'], default='multi_retrieval', help='Pipeline type')

    # Prompt Config
    parser.add_argument('--prompt_set', type=int, default=26, help='Prompt set')
    parser.add_argument('--prompt_document_from', choices=['last_only', 'full'], default='last_only', help='Document selection for prompts')
    parser.add_argument('--prompt_max_para_count', type=int, default=15, help='Max paragraph count')
    parser.add_argument('--prompt_max_para_words', type=int, default=350, help='Max words per paragraph')

    # Generation Config
    parser.add_argument('--generation_model_name', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='Model name for generation')
    parser.add_argument('--generation_max_batch_size', type=int, default=16, help='Max batch size for generation') # 16 for 80GB, sometimes OOM...
    parser.add_argument('--generation_max_total_tokens', type=int, default=4096, help='Max total tokens for generation')
    parser.add_argument('--generation_max_new_tokens', type=int, default=64, help='Max new tokens for generation')
    parser.add_argument('--generation_min_new_tokens', type=int, default=1, help='Min new tokens for generation')

    # Retrieval Config
    parser.add_argument('--retrieval_query_type', choices=['last_only', 'full'], default='full', help='Query type for retrieval')
    parser.add_argument('--retrieval_count', choices=[2, 4, 6, 8], default=8, help='Number of retrievals')
    parser.add_argument('--retrieval_buffer_size', type=int, default=32, help='Buffer size for retrieval')
    parser.add_argument('--retrieval_no_duplicates', action='store_true', help='No duplicate retrievals')
    parser.add_argument('--retrieval_no_reasoning_sentences', action='store_true', help='Exclude reasoning sentences from retrieval')
    parser.add_argument('--retrieval_no_wh_words', action='store_true', help='Exclude WH words from retrieval')

    # Retriever Config
    parser.add_argument('--retrieval_query_model_name_or_path', type=str, default='facebook/contriever-msmarco', help='Retrieval query model path')
    parser.add_argument('--retrieval_passage_model_name_or_path', type=str, default=None, help='Retrieval passage model path')
    parser.add_argument('--retrieval_batch_size', type=int, default=32, help='Batch size for retrieval')
    parser.add_argument('--retrieval_training_strategy', choices=['query_only', 'both'], default='query_only', help='Training strategy for retrieval')
    parser.add_argument('--retrieval_use_fp16', action='store_true', help='Use FP16 in retrieval')

    # End Config
    parser.add_argument('--max_num_thought', type=int, default=6, help='Max number of thoughts')
    parser.add_argument('--answer_regex', type=str, default=".* Answer: <.*>\\.?", help='Regex for answer matching')
    parser.add_argument('--match_all_on_failure', action='store_true', help='Match all on failure')

    # Etc Config
    parser.add_argument('--demo', action='store_true', help='Run in demo mode')

    # Training Config
    parser.add_argument('--train', default=True, help='Enable training mode')
    parser.add_argument('--n_epochs', type=int, default=3, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--temperature_r', type=float, default=0.1, help='Temperature for retrieval')
    parser.add_argument('--temperature_lm', type=float, default=1.0, help='Temperature for language model')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--wandb_key', type=str, default=None, help='WandB API key')

    args = parser.parse_args()
    
    return PipelineConfig(**vars(args))

def get_pipeline(cfg, contexts, generator, retriever, indexer):
    if cfg.pipeline_type == 'no_retrieval':
        raise NotImplementedError('...')
        
    elif cfg.pipeline_type == 'single_retrieval':
        raise NotImplementedError('...')

    elif cfg.pipeline_type == 'multi_retrieval':
        pipeline = [
            RetrievalStep(
                cfg=cfg,
                retriever=retriever,
                indexer=indexer,
                # contexts=contexts,
            ),
            TrainStep(
            cfg,
            generator=generator,
            retriever=retriever,
            indexer=indexer,
            ),
            GenerationStep(
                cfg=cfg,
                generator=generator,
                prompt_generator=AnswerGeneratePromptGenerator(cfg),
                output_parser=AnswerGenerateOutputParser(cfg)
            ),
            EndStep(
                cfg=cfg,
            ),
            GenerationStep(
                cfg=cfg,
                generator=generator,
                prompt_generator=ThoughtGeneratePromptGenerator(cfg),
                output_parser=ThoughtGenerateOutputParser(cfg)
            ),
        ]
    
    return pipeline

def validate(controller, epoch, num_steps):
    total_loss = 0
    total_batches = 0

    controller.pipeline[0].retriever.query_model.eval() 

    cfg.dataset_split = "dev"

    dev_inputs, dev_id_to_ground_truths, dev_contexts = load_data_from_jsonl(
        file_path = cfg.data_file_path,
        ground_truth_file_path=cfg.id_to_ground_truths_file_path,
        return_contexts=True,
        is_demo=opt.demo
    )
    dev_start_states = [
        QuestionState(
            question_id=question_id,
            question=question_text,
            answer=dev_id_to_ground_truths[question_id][0],  # Ground truth answer for trainstep
        )
        for question_id, question_text in dev_inputs.items()
    ]

    dev_dataloader = DataLoader(
        QuestionStateDataset(dev_start_states),
        batch_size=cfg.batch_size, 
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_question_states,
    )
    with torch.no_grad():  # Disable gradient calculation for validation
        for batch in dev_dataloader:
            batch_loss = controller.train(batch)
            total_loss += batch_loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches
    print(f"Epoch {epoch}, Step {num_steps}: Validation Loss {avg_loss}")

    controller.pipeline[0].retriever.query_model.train()

def train(cfg, generator, retriever, indexer, optimizer, scheduler):
    cfg.dataset_split = "train"

    clean_and_create_dir(cfg.prediction_file_dir)
    cfg.save()

    inputs, id_to_ground_truths, contexts = load_data_from_jsonl(
        file_path = cfg.data_file_path,
        ground_truth_file_path=cfg.ground_truth_file_path,
        return_contexts=True,
        is_demo=cfg.demo,
    )
    
    controller = PipelineController(
        pipeline=get_pipeline(cfg, contexts, generator, retriever, indexer),
        # cfg=cfg,
        # id_to_ground_truths=id_to_ground_truths,
        logging_file_path=cfg.logging_file_path,
        prediction_file_path=cfg.prediction_file_path,
    )

    start_states = [
        QuestionState(
            question_id=question_id,
            question=question_text,
            answer=id_to_ground_truths[question_id][0],  # Ground truth answer
        )
        for question_id, question_text in inputs.items()
    ]

    dataloader = DataLoader(
        QuestionStateDataset(start_states),
        batch_size=cfg.batch_size, 
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_question_states,
    )

    scheduler = ExponentialLR(optimizer, gamma=0.9) 
    num_accumulations = 0
    validation_freq = 10

    for epoch in range(cfg.n_epochs):
        num_steps = 0

        for batch in dataloader:
            batch_loss = controller.train(batch)
            batch_loss.backward()

            num_accumulations += 1
            num_steps += 1

            if num_accumulations % cfg.gradient_accumulation_steps == 0:
                num_accumulations = 0
                optimizer.step()
                optimizer.zero_grad()

                if num_steps % 100 == 0:
                    if scheduler:
                        scheduler.step()
                    
            if cfg.wandb_key:
                log = {
                    f'training loss': batch_loss.item(), 
                    f'epoch': epoch,
                    f'step': num_steps,
                    }
                if (scheduler is not None):
                    log["learning_rate"] = scheduler.get_last_lr()[0]
                wandb.log(log)

            print(f"Step {num_steps}: Training Loss {batch_loss} logging!")

            if num_steps % validation_freq == 0:
                validate(controller, epoch, num_steps)
            
            if num_steps % 100 == 0:
                save_path = os.path.join(cfg.prediction_file_dir, f"epoch_{epoch}_step_{num_steps}")
                retriever.query_model.save_pretrained(save_path)
                retriever.query_tokenizer.save_pretrained(save_path)
                print(f"Retriever saved in {save_path} at epoch_{epoch}_step_{num_steps}!")

    save_path = cfg.prediction_file_dir
    retriever.query_model.save_pretrained(save_path)
    retriever.query_tokenizer.save_pretrained(save_path)
    print(f"Final retriever saved in {save_path}!")


if __name__ == '__main__':
    opt = parse_args()
    cfg = PipelineConfig(**opt.__dict__)

    generator = LlamaGenerator(
        LlamaGeneratorConfig(
        model_name=opt.generation_model_name,
        batch_size=opt.generation_max_batch_size,
        max_total_tokens=opt.generation_max_total_tokens,
        max_new_tokens=opt.generation_max_new_tokens,
        min_new_tokens=opt.generation_min_new_tokens,
        use_vllm=False,
        )
    )
    retriever = DenseRetriever(
        DenseRetrieverConfig(
            query_model_name_or_path=opt.retrieval_query_model_name_or_path,
            passage_model_name_or_path=opt.retrieval_passage_model_name_or_path,
            batch_size=opt.retrieval_batch_size,
            training_strategy=opt.retrieval_training_strategy,
            use_fp16=opt.retrieval_use_fp16,
        )
    )
    indexer = Indexer.load_local(
        IndexerConfig(
            embedding_sz=768,
            database_path= cfg.database_path
        )
    )
    
    optimizer = AdamW(
        retriever.query_model.parameters(), 
        lr=cfg.lr
    )

    scheduler = None # scheduler is defined in the train function, too lazy to fix this...

    train(cfg, generator, retriever, indexer, optimizer, scheduler)
