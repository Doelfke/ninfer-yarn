# NInfer (YaRN)

This is a fork of [Ninfer](https://github.com/gzenz/ninfer) that adds YaRN context extension via
`--rope-scaling-factor` and `--rope-scaling-original-context`.

## YaRN context extension

The registered models have a native context limit of 262,144 tokens. YaRN linear position scaling
extends it: positions at or below `--rope-scaling-original-context` (default 262144) are unchanged,
and larger positions map to `original_context + (position - original_context) / factor`. The
effective `--max-context` ceiling becomes `min(native * factor, 8388608)`, so a factor of 2.12
admits roughly 555k-token sequences.

- `--rope-scaling-factor F`: YaRN factor in `1.0..32.0`; `1.0` (the default) disables scaling.
- `--rope-scaling-original-context N`: ramp threshold, default `262144`.
- The flags are accepted by `ninfer` (CLI), `ninfer-perplexity`, and `ninfer-serve`.
- Not supported with masked-draft speculative decoding (`--spec dflash` or `--spec dflash2`).
- Longer contexts still need enough KV capacity: prefer a quantized `--kv-dtype`
  (`nvfp4`/`k8v4`/`fp8`) and size `--kv-capacity` (or `auto`) accordingly.

Example:

```bash
./build/apps/ninfer-serve <model.ninfer> \
   --host 0.0.0.0 --port 8080 \
   --max-context 555000 --kv-dtype nvfp4 \
   --spec mtp --draft-tokens 5 --lm-head-draft \
   --rope-scaling-factor 2.12 --rope-scaling-original-context 262144
```

See [docs/cli.md](docs/cli.md) and [docs/serving.md](docs/serving.md) for the full option tables.