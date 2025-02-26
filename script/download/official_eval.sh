# This script is based on code from the IRCOT project by Stony Brook NLP.
# Source: https://github.com/StonyBrookNLP/ircot

# mkdir -p source/metrics

git clone https://github.com/hotpotqa/hotpot source/evaluation/official_evaluation/hotpotqa
cd source/evaluation/metrics/official_evaluation/hotpotqa ; git checkout 3635853403a8735609ee997664e1528f4480762a
cd ../../../../

git clone https://github.com/Alab-NII/2wikimultihop source/evaluation/official_evaluation/2wikimultihopqa
cd source/evaluation/official_evaluation/2wikimultihopqa ; git checkout 6bdd033bd51aae2d36ba939688c651b5c54ec28a
cd ../../../../

git clone https://github.com/stonybrooknlp/musique source/evaluation/official_evaluation/musique
cd source/evaluation/official_evaluation/musique ; git checkout 24cc5b297acc2abfc5fb3d0becb6ef7b73d03717
cd ../../../../
