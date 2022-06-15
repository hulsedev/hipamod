module load python/3.7
module load cuda/10.2/cudnn/7.6
module load cudatoolkit/10.2
source $HOME/live/bin/activate
pip install -r requirements.txt
FORCE_CUDA=1 TOKENIZERS_PARALLELISM=true python3 scripts/eval_gptneo125m.py
