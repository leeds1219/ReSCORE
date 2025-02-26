from source.utility.data_utils import (
    load_data_from_jsonl
)
from source.pipeline.step.__retrieval import RetrievalStep
from source.pipeline.step.generation import (
    GenerationStep, 
    AnswerGenerateOutputParser, 
    AnswerGeneratePromptGenerator,
    ThoughtGenerateOutputParser,
    ThoughtGeneratePromptGenerator,
)
from source.pipeline.step.end import EndStep

from source.pipeline.config import PipelineConfig
from source.pipeline.controller import PipelineController
from source.pipeline.state import QuestionState
import os
from source.utility.system_utils import seed_everything
from source.utility.data_utils import clean_and_create_dir

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
import json
from source.evaluation.evaluate import (
    evaluate_by_dicts,
    official_evaluate_by_dicts,
)
import copy

def run(cfg, generator, retriever, indexer):
    clean_and_create_dir(cfg.prediction_file_dir)
    cfg.save()

    inputs, id_to_ground_truths, contexts = load_data_from_jsonl(
        file_path = cfg.data_file_path,
        ground_truth_file_path=cfg.ground_truth_file_path,
        return_contexts=True,
        is_demo=opt.demo,
    )

    pipeline = [
        RetrievalStep(
            cfg=cfg,
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
    controller = PipelineController(
        pipeline=pipeline,
        logging_file_path=cfg.logging_file_path,
        prediction_file_path=cfg.prediction_file_path,
    )
    start_states = [
        QuestionState(
            question_id=question_id,
            question=question_text
        )
        for question_id, question_text in inputs.items()
    ]
    controller.run(
        start_states,
        batch_size=cfg.batch_size
    )
    
    with open(cfg.ground_truth_file_path, 'r', encoding='utf-8') as f:
        id_to_ground_truths = json.load(f)
        
    with open(cfg.prediction_file_path, 'r', encoding='utf-8') as f:
        id_to_predictions = json.load(f)
        
    evaluation_results = evaluate_by_dicts(
        prediction_type='answer',
        id_to_ground_truths=id_to_ground_truths,
        id_to_predictions=id_to_predictions,
    )
    with open(cfg.evaluation_file_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f)
    official_evaluation_results = official_evaluate_by_dicts(
        prediction_type='answer',
        id_to_ground_truths=id_to_ground_truths,
        id_to_predictions=id_to_predictions,
        dataset=cfg.dataset
    )
    with open(cfg.official_evaluation_file_path, 'w') as f:
        json.dump(official_evaluation_results, f)
    
    return official_evaluation_results['f1']


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # General parameters
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="iqatr or base"
    )    
    parser.add_argument(
        "--running_name",
        type=str,
        default=None,
        help=""
    )    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Inference Batch size"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Random seed"
    )
    parser.add_argument(
        "--dataset",
        choices=['hotpotqa', '2wikimultihopqa', 'musique'],
        default='musique',
        help="Dataset name"
    )
    parser.add_argument(
        "--prompt_set",
        type=int,
        default=1,
        help="prompt_set"
    )
    parser.add_argument(
        "--prompt_document_from",
        choices=['last_only', 'full'],
        default='last_only',
        help="prompt_document_from"
    )
    parser.add_argument(
        "--prompt_max_para_count",
        type=int,
        default=15,
        help="Maximum number of paragraphs in prompt"
    )
    parser.add_argument(
        "--prompt_max_para_words",
        type=int,
        default=350,
        help="Maximum words per paragraph in prompt"
    )

    # Generator
    parser.add_argument(
        "--generation_model_name",
        type=str,
        default='meta-llama/Meta-Llama-3.1-8B-Instruct',
        help="Generation model name"
    )
    parser.add_argument(
        "--generation_max_batch_size",
        type=int,
        default=4,
        help="Maximum batch size for generation"
    )
    parser.add_argument(
        "--generation_max_total_tokens",
        type=int,
        default=4096,
        help="Maximum total tokens for generation"
    )
    parser.add_argument(
        "--generation_max_new_tokens",
        type=int,
        default=64,
        help="Maximum new tokens for generation"
    )
    parser.add_argument(
        "--generation_min_new_tokens",
        type=int,
        default=1,
        help="Minimum new tokens for generation"
    )

    # Retrieval
    parser.add_argument(
        "--retrieval_count",
        type=int,
        choices=[2, 4, 6, 8],
        default=8,
        help="Number of retrievals"
    )
    parser.add_argument(
        "--retrieval_query_type",
        choices=['last_only', 'full'],
        default='full',
        help="Retrieval Query type"
    )
    parser.add_argument(
        "--retrieval_buffer_size",
        type=int,
        default=32,
        help="Retrieval buffer size"
    )
    parser.add_argument(
        "--retrieval_no_duplicates",
        action='store_true',
        help="Remove duplicate retrievals"
    )
    parser.add_argument(
        "--retrieval_no_reasoning_sentences",
        action='store_true',
        help="Exclude reasoning sentences from retrieval"
    )
    parser.add_argument(
        "--retrieval_no_wh_words",
        action='store_true',
        help="Exclude WH-words from retrieval"
    )

    # Retriever
    parser.add_argument(
        "--retrieval_query_model_name_or_path",
        type=str,
        default='facebook/contriever-msmarco',
        help="Query model name or path for retrieval"
    )
    parser.add_argument(
        "--retrieval_passage_model_name_or_path",
        type=str,
        default=None,
        help="Passage model name or path for retrieval"
    )
    parser.add_argument(
        "--retrieval_batch_size",
        type=int,
        default=32,
        help="Batch size for retrieval"
    )
    parser.add_argument(
        "--retrieval_training_strategy",
        choices=['query_only', 'both'],
        default=None,
        help="Training strategy for retrieval"
    )
    parser.add_argument(
        "--retrieval_use_fp16",
        action='store_true',
        help="Use FP16 for retrieval"
    )

    # End
    parser.add_argument(
        "--max_num_thought",
        type=int,
        default=6,
        help="Maximum number of thoughts"
    )
    parser.add_argument(
        "--answer_regex",
        type=str,
        default=".* answer is:? (.*)\\.?",
        help="Regex pattern to extract answer"
    )
    
    # Etc
    parser.add_argument(
        "--demo",
        action='store_true',
        help="Whether to use Demo"
    )

    opt = parser.parse_args()

    seed_everything(opt.seed)
    
    cfg = PipelineConfig(**opt.__dict__)

    generator = LlamaGenerator(
        LlamaGeneratorConfig(
        model_name=opt.generation_model_name,
        batch_size=opt.generation_max_batch_size,
        max_total_tokens=opt.generation_max_total_tokens,
        max_new_tokens=opt.generation_max_new_tokens,
        min_new_tokens=opt.generation_min_new_tokens,
        use_vllm=True,
        # use_vllm=False,
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

    cfg.dataset_split = 'test'
    run(cfg, generator, retriever, indexer)
