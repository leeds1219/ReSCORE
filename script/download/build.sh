python -m source.run.preprocess_raw_data \
--dataset_name hotpotqa; \
python -m source.run.generate_passage_embeddings \
--model_name_or_path facebook/contriever-msmarco \
--passages ./data/embed_ready_data/hotpotqa.tsv \
--output_dir ./data/database/contriever_msmarco/hotpotqa \
--shard_id 0 \
--num_shards 1 \
--per_gpu_batch_size 1024; \
python -m source.run.build_index \
--output_dir ./data/database/contriever_msmarco/hotpotqa; \


python -m source.run.preprocess_raw_data \
--dataset_name musique; \
python -m source.run.generate_passage_embeddings \
--model_name_or_path facebook/contriever-msmarco \
--passages ./data/embed_ready_data/musique.tsv \
--output_dir ./data/database/contriever_msmarco/musique \
--shard_id 0 \
--num_shards 1 \
--per_gpu_batch_size 1024; \
python -m source.run.build_index \
--output_dir ./data/database/contriever_msmarco/musique; \


python -m source.run.preprocess_raw_data \
--dataset_name 2wikimultihopqa; \
python -m source.run.generate_passage_embeddings \
--model_name_or_path facebook/contriever-msmarco \
--passages ./data/embed_ready_data/2wikimultihopqa.tsv \
--output_dir ./data/database/contriever_msmarco/2wikimultihopqa \
--shard_id 0 \
--num_shards 1 \
--per_gpu_batch_size 1024; \
python -m source.run.build_index \
--output_dir ./data/database/contriever_msmarco/2wikimultihopqa
