# NInfer (YaRN)

This is a fork of [Ninfer](https://github.com/gzenz/ninfer), an inference engine designed to be optimized for Qwen3.x models and an RTX 5090. This project adds YaRN context extension and other enhancements:

- `--rope-scaling-factor` and `--rope-scaling-original-context` 
- `--tolerant-tool-calls` for Qwen, to ensure tool calls are OpenAI compatible.  
- `--vision-cpu` to offload vision.  

When using an NVFP4 KV cache, this allows you to reach a 420,000 token context, while having vision enabled, MTP -- supporting 2 sessions at once.  This assumes `maxOutputTokens` is set to 130,000 in your code editor.



Example:

```bash

services:
  ninfer:
    image: ninfer:local
    command: [
        "ninfer-serve", 
        "/models/qwen3_8_27b_nvfp4.ninfer", 
        "--host", "0.0.0.0",
        "--max-context", "420000",
        "--kv-dtype", "nvfp4",
        "--kv-capacity", "420000",
        "--max-concurrency", "2",
        "--spec", "mtp",
        "--draft-tokens", "5",
        "--lm-head-draft",
        "--preserve-thinking",
        "--tolerant-tool-calls",
        "--rope-scaling-factor", "2",
        "--rope-scaling-original-context", "262144",
        "--vision-cpu",
        "--pending-timeout-ms", "300000",
        "--device-state-slots", "2",
        "--host-state-slots", "8",
        "--host-kv-mib", "8192"
      ]
    ports:
      - "1234:8080"
    volumes:
      - /home/x/models:/models:ro
      
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]

```