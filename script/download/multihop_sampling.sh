# # Process raw data files in a single standard format
# python ./preprocess/process_nq.py

# python ./preprocess/process_trivia.py

# python ./preprocess/process_squad.py

# Subsample the processed datasets
python ./preprocess/subsample_dataset_and_remap_paras.py --dataset_name musique --set_name dev
python ./preprocess/subsample_dataset_and_remap_paras.py --dataset_name musique --set_name test
python ./preprocess/subsample_dataset_and_remap_paras.py --dataset_name musique --set_name dev_diff_size --sample_size 1000

python ./preprocess/subsample_dataset_and_remap_paras.py --dataset_name 2wikimultihopqa --set_name dev
python ./preprocess/subsample_dataset_and_remap_paras.py --dataset_name 2wikimultihopqa --set_name test
python ./preprocess/subsample_dataset_and_remap_paras.py --dataset_name 2wikimultihopqa --set_name dev_diff_size --sample_size 1000

python ./preprocess/subsample_dataset_and_remap_paras.py --dataset_name hotpotqa --set_name dev
python ./preprocess/subsample_dataset_and_remap_paras.py --dataset_name hotpotqa --set_name test
python ./preprocess/subsample_dataset_and_remap_paras.py --dataset_name hotpotqa --set_name dev_diff_size --sample_size 1000