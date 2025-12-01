# LeakAgent: RL-based Red-teaming Agent for LLM Privacy Leakage

## How to run (without GPU)

### 1. Install the requirements

```bash
pip install -r requirements.txt
```

### 2. train the model

launch [Flask app](https://github.com/arturo-b-cmu/pplt-project) for target model


```shell
# target_model can be set to anything that includes 'gpt'
python3 pipeline.py \
    --method fuzz \
    --target_model medical_data_gpt \
    --target_server_url http://127.0.0.1:8081/v1 \
    --target_api_key test \
    --prompts_data_path train_data_pleak.csv
```

### 3. Evaluate the model

launch [Flask app](https://github.com/arturo-b-cmu/pplt-project) for target model

for evaluating the model, you can run the following command:

```shell
# evaluate
python evaluate_task.py \
            --model_name medical_data \
            --prompts_data_path /your/output_dir/good_prompts.csv \
            --n_samples 10 \
            --server_url http://127.0.0.1:8081/v1 \
            --api_key test \
            --dataset_path test_data_pleak.csv
```


### 4. Reference
If you find this repository useful, please consider citing the following paper:

```bibtex
@inproceedings{nie2025leakagent,
      title={LeakAgent: RL-based Red-teaming Agent for LLM Privacy Leakage}, 
      author={Yuzhou Nie and Zhun Wang and Ye Yu and Xian Wu and Xuandong Zhao and Wenbo Guo and Dawn Song},
      year={2025},
      booktitle={COLM}, 
}

```
