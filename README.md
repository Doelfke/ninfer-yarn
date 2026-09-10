# NInfer (YaRN)

This is a fork of [Ninfer](https://github.com/gzenz/ninfer) that adds YaRN context extension via
`--rope-scaling-factor` and `--rope-scaling-original-context`.

Adds `--tolerant-tool-calls` for Qwen, to ensure tool calls are OpenAI compatible.  

Adds --vision-cpu to offload vision.  

When using an NVFP4 KV cache, this allows you to reach a 450,000 context, while having vision enabled.


Example:

```bash
./build/apps/ninfer-serve <model.ninfer> \
   --host 0.0.0.0 --port 8080 \
   --max-context 450000 --kv-dtype nvfp4 \
   --spec mtp --draft-tokens 5 --lm-head-draft \
   --rope-scaling-factor 2 --rope-scaling-original-context 262144 \
   --tolerant-tool-calls
   --vision-cpu
```