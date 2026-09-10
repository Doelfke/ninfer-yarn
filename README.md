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

## `--vision-cpu (WIP)`

`--vision-cpu` is `--vision` with the Vision encoder (ViT) running on CPU: the backbone and merger
weights are dequantized once at load into host DRAM rather than the GPU arena, each multimodal
item is encoded on the host, and the projected embedding is handed to the device with a single
copy. Use it to free the encoder's GPU footprint when VRAM is the constraint and Vision encode
throughput is not. The freed size is reported as `host_vision_weights_bytes` in the memory summary
(the device `encode_peak_bytes` is `0` and the only device-resident Vision region is the small
embedding handoff).
