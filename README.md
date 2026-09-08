# NInfer (YaRN)

This is a fork of [Ninfer](https://github.com/gzenz/ninfer) that adds YaRN context extension via
`--rope-scaling-factor` and `--rope-scaling-original-context`.

It also adds in `--tolerant-tool-calls` for Qwen.

The code was ported from [gzenz](https://github.com/gzenz/ninfer), which targets more changes.  This is a minimal fork to add YaRN support and Qwen tool calling fixes.

Example:

```bash
./build/apps/ninfer-serve <model.ninfer> \
   --host 0.0.0.0 --port 8080 \
   --max-context 400000 --kv-dtype nvfp4 \
   --spec mtp --draft-tokens 5 --lm-head-draft \
   --rope-scaling-factor 1.6 --rope-scaling-original-context 262144 \
   --tolerant-tool-calls
```
