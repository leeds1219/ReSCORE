import gradio as gr
from source.pipeline.config import PipelineConfig
from source.pipeline.controller import PipelineController
from source.pipeline.step.retrieval import RetrievalStep
from source.pipeline.step.generation import (
    GenerationStep, 
    AnswerGenerateOutputParser, 
    AnswerGeneratePromptGenerator,
    ThoughtGenerateOutputParser,
    ThoughtGeneratePromptGenerator,
)
from source.pipeline.step.end import EndStep
from source.pipeline.state import QuestionState
from source.module.generate.llama import LlamaGenerator, LlamaGeneratorConfig
from source.module.retrieve.dense import DenseRetriever, DenseRetrieverConfig
from source.module.index.index import Indexer, IndexerConfig
from source.utility.system_utils import seed_everything

from huggingface_hub import login

login(token=f"{your_hf_token}")

seed_everything(100)

cfg = PipelineConfig(
    # 필요한 최소 설정
    method="base",
    batch_size=1,
    generation_model_name='meta-llama/Meta-Llama-3.1-8B-Instruct',
    generation_max_batch_size=1,
    generation_max_total_tokens=4096,
    generation_max_new_tokens=64,
    generation_min_new_tokens=1,
    retrieval_count=8,
    retrieval_query_type='full',
    dataset='musique',
    max_num_thought=6,
    answer_regex=".* answer is:? (.*)\\.?"
)

generator = LlamaGenerator(
    LlamaGeneratorConfig(
        model_name=cfg.generation_model_name,
        batch_size=cfg.generation_max_batch_size,
        max_total_tokens=cfg.generation_max_total_tokens,
        max_new_tokens=cfg.generation_max_new_tokens,
        min_new_tokens=cfg.generation_min_new_tokens,
        use_vllm=False, #True
        gpu=0,
    )
)

retriever = DenseRetriever(
    DenseRetrieverConfig(
        query_model_name_or_path='facebook/contriever-msmarco',
        passage_model_name_or_path=None,
        batch_size=32,
        training_strategy=None,
        use_fp16=False
    )
)

indexer = Indexer.load_local(
    IndexerConfig(
        embedding_sz=768,
        database_path=cfg.database_path
    )
)

pipeline = [
    RetrievalStep(cfg=cfg, retriever=retriever, indexer=indexer),
    GenerationStep(cfg=cfg, generator=generator,
                   prompt_generator=AnswerGeneratePromptGenerator(cfg),
                   output_parser=AnswerGenerateOutputParser(cfg)),
    EndStep(cfg=cfg),
    GenerationStep(cfg=cfg, generator=generator,
                   prompt_generator=ThoughtGeneratePromptGenerator(cfg),
                   output_parser=ThoughtGenerateOutputParser(cfg)),
]

controller = PipelineController(
    pipeline=pipeline,
    logging_file_path=None,
    prediction_file_path=None
)

# # ----------------------------------------------------------
# # Gradio UI
# # ----------------------------------------------------------
import gradio as gr


def run_pipeline(user_input):
    global controller
    global QuestionState

    logs = []

    # QuestionState 생성
    start_state = QuestionState(question_id="1", question=user_input)

    # ---------------------- HOP LOOP ----------------------
    # ---- 1-hop (special case) ----
    controller.update([start_state])
    paths = controller.next()

    # Retrieve
    next_states = controller.pipeline[0](paths)
    titles = [doc.metadata['title'] for doc in next_states[0].documents[:8]]
    logs.append("1-hop Retrieve:" + "\n".join(titles))

    controller.update(next_states)
    paths = controller.next()

    # Answer
    next_states = controller.pipeline[1](paths)
    logs.append(f"1-hop Answer: {next_states[0].answer}")

    # TODO: We found that the constraint of at least 2-hops benefits.
    if next_states[0].answer != "Unknown":
        logs.append("1-hop answer obtained, proceeding with an additional hop for verification.")
        next_states[0].answer = "Unknown"
        # return "\n\n".join(logs)
    
    controller.update(next_states)
    paths = controller.next()

    # Check
    next_states = controller.pipeline[2](paths)
    controller.update(next_states)
    paths = controller.next()

    # Think
    next_states = controller.pipeline[3](paths)
    logs.append(f"1-hop Thought: {next_states[0].thought}")

    controller.update(next_states)
    paths = controller.next()

    # ---- Loop for hop >= 2 ----
    MAX_HOPS = 6
    hop = 2

    while hop <= MAX_HOPS:
        # Retrieve
        next_states = controller.pipeline[0](paths)
        # TODO: we use a buffer of 32 to remove redundant documents, this is not implemented in this demo
        titles = [doc.metadata['title'] for doc in next_states[0].documents[:8]]
        logs.append(f"{hop}-hop Retrieve:" + "\n".join(titles))

        controller.update(next_states)
        paths = controller.next()

        # Answer
        next_states = controller.pipeline[1](paths)
        logs.append(f"{hop}-hop Answer: {next_states[0].answer}")

        if next_states[0].answer != "Unknown":
            break

        controller.update(next_states)
        paths = controller.next()

        # Check
        next_states = controller.pipeline[2](paths)
        controller.update(next_states)
        paths = controller.next()

        # Think
        next_states = controller.pipeline[3](paths)
        logs.append(f"{hop}-hop Thought: {next_states[0].thought}")

        controller.update(next_states)
        paths = controller.next()

        hop += 1

    # ------------------------------------------------------------
    return "\n\n".join(logs)


# ====================== Gradio UI ==============================

def demo_ui():
    with gr.Blocks() as demo:

        gr.Markdown("# Multi-Hop Retrieval Pipeline Demo")

        question_box = gr.Textbox(
            label="Question",
            value="Where was the author of Hannibal and Scipio educated at?",
            lines=2,
        )

        """
        Examples)

        Which company owns the manufacturer of Learjet 60?

        In which county is Southern Maryland Electric Cooperative headquartered?

        What is another notable work made by the author of Miss Sara Sampson?

        What is the seat of the county where Van Hook Township is located?

        The Unwinding author volunteered for which organisation?

        ...
        """
        
        output_box = gr.Textbox(
            label="Logs",
            lines=40,
        )

        run_btn = gr.Button("Run Pipeline")

        run_btn.click(
            fn=run_pipeline,
            inputs=question_box,
            outputs=output_box,
        )

    return demo


if __name__ == "__main__":
    demo = demo_ui()
    demo.launch(share=True)

