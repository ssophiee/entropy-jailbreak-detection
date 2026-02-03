Example for running the computation on GPU, in full precision, on Qwen2.5-3B-Instruct. 

```shell
python -m approaches.semantic_entropy.compute_semantic_entropy \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --n_samples 25 \
  --max_new_tokens 128 \
  --temperature 0.9 \
  --top_p 0.95 \
  --cluster_threshold 0.82 \
  --embed_model_name sentence-transformers/all-MiniLM-L6-v2 \
  --embed_batch_size 64 \
  # --fp16 \
  --output_dir results/semantic_entropy/Qwen2.5_3B_Instruct
```

Then, to evaluate it: 

```shell
python approaches/semantic_entropy/eval_semantic_entropy.py PATH_TO_JSON_RESULT_FILE
```