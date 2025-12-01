# LeakAgent: RL-based Red-teaming Agent for LLM Privacy Leakage

## How to run

### Install the requirements

```bash
pip install -r requirements.txt
```
```bash
python3 - <<'PY'
import nltk
nltk.download('punkt_tab')
PY
```

## Attack Options

### Sentence-level attack generation (CPU-only)
- mutates and evolves a seed prompt to find better ones
- good for quick testing and finding prompt injection vectors

1. Launch [Flask app](https://github.com/arturo-b-cmu/pplt-project) for target model

2. Run pipeline.py
```shell
python3 pipeline.py \
  --method fuzz \
  --target_model gpt_local \
  --prompts_data_path train_data_pleak.csv \
  2>&1 | tee pipeline_output.log
```
- Methods
    - fuzz: Fuzzing-based search (UCB/MCTS seed selection + mutations)
    - re: ReAct (reasoning-based attack generation)
    - sent_rl: Lightweight sentence-level RL
    - gcg / prob: Token-level gradient/probabilistic methods

### RL Fine-Tuning of Large Model (GPU)
- fine-tunes a LLM using PPO and RLHF
- Uses a victim model to score/reward candidate prompts
- Output: fine-tuned model checkpoint + good prompts discovered during training

1. Launch [Flask app](https://github.com/arturo-b-cmu/pplt-project) for target model

2. Train the model
   
```shell
accelerate launch --multi_gpu --num_machines 1 --num_processes 2 --main_process_port 24599 \
attacks/token_level/blackbox/FineTuneLLM.py \
    --model_name=meta-llama/Meta-Llama-3-8B-Instruct --adafactor=False \
    --tokenizer_name=meta-llama/Meta-Llama-3-8B-Instruct --load_in_4bit \
    --victim_model meta-llama/Llama-3.1-8B-Instruct \
    --output_dir llama3_rl_finetune/ --batch_size 32 \
    --requests_per_minute 100 --target_dataset train_data_pleak.csv \
    --epochs 60 --save_freq 4 \
    --use_bonus_reawrd True \
    --server_url http://127.0.0.1:8081/v1 --api_key test
```

3. Evaluate the model

```shell
python evaluate_task.py \
  --model_name medical_data \
  --prompts_data_path /your/output_dir/good_prompts.csv \
  --n_samples 10 \
  --server_url http://127.0.0.1:8081/v1 \
  --api_key test \
  --dataset_path test_data_pleak.csv
```


### Reference
Original Paper

```bibtex
@inproceedings{nie2025leakagent,
      title={LeakAgent: RL-based Red-teaming Agent for LLM Privacy Leakage}, 
      author={Yuzhou Nie and Zhun Wang and Ye Yu and Xian Wu and Xuandong Zhao and Wenbo Guo and Dawn Song},
      year={2025},
      booktitle={COLM}, 
}

```



