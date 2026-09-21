"""
DFlash2 acceptance benchmark -- a FIXED, reproducible suite for before/after
comparison of acceptance-improvement changes.

This is not a free sweep: the cell set, prompts and sampling are FIXED so that
two runs (before a code change and after) exercise identical inputs and can be
diffed directly with --compare. The server reports timings.draft_n (drafted)
and timings.draft_n_accepted (accepted) per request; no log parsing is needed.

Regimes covered (from live measurement, 2026-09-21):
  - short context       (1K)    -- guardrail: must not regress
  - mid context         (32K)   -- flat baseline
  - just under cliff    (80K)   -- still healthy (~44-47%)
  - just over cliff     (122K)  -- the drop zone (~35-40%); main improvement target
  - temperature 0 and 1         -- isolates the stochastic/entropy penalty (+5-10 pts)

Determinism
-----------
Prompts are sized to exact token counts via the server tokenizer (binary
search, rope_test.py pattern -- guessing chars/token oversizes 20-30x). Seeds
derive from SHA-256 (NOT Python's salted hash()), so the exact same prompts are
rebuilt on every run. Filler is per-cell unique so prompt-cache replay cannot
inflate measurements (a production run showed 86.2% cache replay).

Usage
-----
  # standard suite (4 prompts x T={0,1}, story to fill max_tokens): ~4-5 min
  python dflash2_acceptance_bench.py --json out/after.json

  # quick (3 prompts, T=1 only) for fast iteration
  python dflash2_acceptance_bench.py --quick --json out/after.json

  # compare against the pre-change run
  python dflash2_acceptance_bench.py --json out/after.json --compare out/before.json

  # inspect a saved run without regenerating
  python dflash2_acceptance_bench.py --from-json out/after.json

K is fixed by the server (--spec dflash2 --draft-tokens K); relaunch per K to
benchmark a K change, and pass the same --draft-tokens to both runs.
"""

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request

BASE = "http://localhost:1234"
MODEL = "qwen3.8-27b"          # matches the model id the server reports
API_KEY = "local"             # arbitrary when the server has no --api-key set
UNIT = "The quick brown fox jumps over the lazy dog. "   # 45 chars of filler
REQUEST_TIMEOUT = 1800.0       # s; long reasoning outputs at long context are slow

# Fixed suites. Completion is a story that fills max_tokens (a short answer would
# leave draft_n mostly idle). thinking is ON by default to match production.
STANDARD_PROMPTS = [1024, 32768, 80000, 122000]
QUICK_PROMPTS    = [1024, 80000, 122000]
COMPLETION       = 4096
STANDARD_TEMPS   = [0.0, 1.0]
QUICK_TEMPS      = [1.0]

QUESTION = (
    "\n\nRecall the secret phrase. Then reason carefully, step by step, "
    "and give your final answer as: Answer: <phrase>."
)

# (target_tokens, seed) -> fitted prompt text
_PROMPT_CACHE: dict[tuple[int, int], str] = {}


class ProbeError(Exception):
    pass


def http_json(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise ProbeError(f"{method} {path} -> HTTP {exc.code}: {exc.read()[:400]!r}")
    except TimeoutError:
        raise ProbeError(f"{method} {path} timed out")


def stable_seed(*keys) -> int:
    """Deterministic seed independent of PYTHONHASHSEED, stable across runs."""
    digest = hashlib.sha256("|".join(str(k) for k in keys).encode()).digest()
    return int.from_bytes(digest[:5], "big")   # 40 bits


def story_question(max_tokens: int) -> str:
    # ~4 chars/token target; aim slightly under so the stream keeps writing
    # until it hits max_tokens (a short answer leaves draft_n mostly idle).
    words = max(max_tokens * 3 // 4, 64)
    return (f"\n\nWrite a creative story of about {words} words. "
            "Keep writing continuously; do not stop early.")


def build_prompt(filler_chars: int, needle: str) -> str:
    """Filler of `filler_chars` chars with a needle inserted at the 90% position."""
    body = (UNIT * (filler_chars // len(UNIT) + 1))[:filler_chars]
    pos = int(len(body) * 0.9)
    return body[:pos] + f" The secret phrase is: {needle}." + body[pos:] + QUESTION


def count_tokens(prompt_text: str) -> int:
    """Exact prompt token count, as the engine's tokenizer will see it."""
    body = {"model": MODEL, "system": "Answer concisely.",
            "messages": [{"role": "user", "content": prompt_text}]}
    return int(http_json("POST", "/v1/messages/count_tokens", body)["input_tokens"])


def fit_prompt(target_tokens: int, seed: int) -> str:
    """
    Binary-search filler length so the REAL prompt is <= target_tokens (and as
    close as possible), then build it with a seed-specific needle. The search
    uses a same-length dummy needle, so the measured count equals the final one.
    """
    key = (target_tokens, seed)
    if key in _PROMPT_CACHE:
        return _PROMPT_CACHE[key]
    dummy = "0" * 10                       # same length as every real needle
    lo, hi = 0, max(target_tokens * 5, 1024)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(build_prompt(mid, f"probe-{dummy}-{target_tokens}")) \
                <= target_tokens:
            lo = mid
        else:
            hi = mid - 1
    needle = f"probe-{seed % 0xFFFFFFFFF:010x}-{target_tokens}"
    prompt = build_prompt(lo, needle)
    _PROMPT_CACHE[key] = prompt
    return prompt


def probe(prompt_text: str, target_tokens: int, completion_tokens: int,
          temperature: float, thinking: bool, k: int) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": prompt_text},
        ],
        "max_tokens": max(completion_tokens, 16),
        "temperature": temperature,
        "include_usage": True,
        "enable_thinking": thinking,
    }
    t0 = time.time()
    out = http_json("POST", "/v1/chat/completions", payload)
    wall_s = time.time() - t0
    usage = out.get("usage") or {}
    completion_n = int(usage.get("completion_tokens", 0))
    choices = out.get("choices") or [{}]
    finish = choices[0].get("finish_reason", "?")
    if completion_n <= 0:
        message = choices[0].get("message", {})
        raise ProbeError(
            f"no completion produced (content={str(message.get('content'))!r:.120}); "
            "check --no-thinking vs max_tokens")
    timings = out.get("timings") or {}
    draft_n = int(timings.get("draft_n", 0))
    draft_acc = int(timings.get("draft_n_accepted", 0))
    if draft_n <= 0:
        raise ProbeError("server returned no draft tokens; is it running "
                         "--spec dflash2 --draft-tokens K?")
    return {
        "target_prompt_tokens": target_tokens,
        "prompt_tokens": int(usage.get("prompt_tokens", -1)),
        "completion_tokens": completion_n,
        "requested_completion": max(completion_tokens, 16),
        "finish_reason": finish,
        "short_filled": completion_n * 10 < max(completion_tokens, 16) * 9,
        "temperature": temperature,
        "cached_tokens": int(usage.get("cached_tokens", 0)),
        "wall_seconds": round(wall_s, 2),
        "drafted": draft_n,
        "accepted": draft_acc,
        "acceptance": draft_acc / draft_n,
        # committed tokens per verify round incl. the bonus token
        "tokens_per_round": completion_n / (draft_n / k),
        "decode_tps": float(timings.get("predicted_per_second", 0.0)),
    }


def run_suite(args) -> list[dict]:
    prompts = QUICK_PROMPTS if args.quick else STANDARD_PROMPTS
    temps = QUICK_TEMPS if args.quick else STANDARD_TEMPS
    prompts = args.prompt_tokens if args.prompt_tokens else prompts
    temps = args.temperature if args.temperature else temps
    thinking = not args.no_thinking
    results = []
    for temperature in temps:
        for prompt in prompts:
            seed = stable_seed("dflash2", prompt, COMPLETION, temperature, "story")
            try:
                text = fit_prompt(prompt, seed)
            except ProbeError as exc:
                print(f"[pt={prompt:>6}, T={temperature}] fit ERROR: {exc}")
                continue
            try:
                row = probe(text.replace(QUESTION, story_question(COMPLETION)),
                            prompt, COMPLETION, temperature, thinking, args.draft_tokens)
            except ProbeError as exc:
                print(f"[pt={prompt:>6}, T={temperature}] probe ERROR: {exc}")
                continue
            results.append(row)
            flag = " <short" if row["short_filled"] else ""
            print(f"[pt={prompt:>6}, T={temperature}] prompt {row['prompt_tokens']:>7,} | "
                  f"out {row['completion_tokens']:>5}{flag} ({row['finish_reason']}) | "
                  f"acc {row['accepted']}/{row['drafted']} "
                  f"({100 * row['acceptance']:.1f}%) | "
                  f"t/round {row['tokens_per_round']:.2f} | "
                  f"{row['decode_tps']:.0f} tps | {row['wall_seconds']:.0f}s")
    return results


def score_of(results: list[dict]) -> dict:
    """Composite for before/after: mean tokens/round (speed) + mean acceptance."""
    if not results:
        return {}
    tpr = sum(r["tokens_per_round"] for r in results) / len(results)
    acc = sum(r["acceptance"] for r in results) / len(results)
    tps = sum(r["decode_tps"] for r in results) / len(results)
    return {"tokens_per_round": round(tpr, 3),
            "acceptance": round(acc, 4),
            "decode_tps": round(tps, 2)}


def print_summary(results: list[dict], title: str) -> dict:
    if not results:
        print(f"\n{title}: no completed cells")
        return {}
    order = sorted(results, key=lambda r: (r["target_prompt_tokens"], r["temperature"]))
    print(f"\n--- {title} ---")
    print(f"{'prompt':>9} {'out':>5} {'T':>4} {'accept':>7} {'drafted':>8} "
          f"{'t/round':>8} {'tps':>6}")
    for row in order:
        print(f"{row['target_prompt_tokens']:>9,} {row['completion_tokens']:>5} "
              f"{row['temperature']:>4.1f} {100 * row['acceptance']:>6.1f}% "
              f"{row['drafted']:>8,} {row['tokens_per_round']:>8.2f} "
              f"{row['decode_tps']:>6.0f}")
    score = score_of(results)
    print(f"SCORE  t/round={score['tokens_per_round']:.3f}  "
          f"accept={100 * score['acceptance']:.1f}%  tps={score['decode_tps']:.0f}")
    return score


def compare(a: list[dict], b: list[dict]) -> None:
    """Row-wise diff of two runs keyed by (prompt, temperature)."""
    key = lambda r: (r.get("target_prompt_tokens"), r.get("temperature"))
    bmap = {key(r): r for r in b}
    print("\n=== compare (A=this run, B=--compare baseline) ===")
    print(f"{'prompt':>9} {'T':>4} {'B acc':>7} {'A acc':>7} {'d acc':>7} "
          f"{'B t/rd':>7} {'A t/rd':>7} {'d t/rd':>7}")
    for row in sorted(a, key=key):
        other = bmap.get(key(row))
        if not other:
            print(f"{row['target_prompt_tokens']:>9,} {row['temperature']:>4.1f}  "
                  f"{'-':>7} {100 * row['acceptance']:>6.1f}% {'(new)':>7}  "
                  f"{'-':>7} {row['tokens_per_round']:>7.2f} {'(new)':>7}")
            continue
        print(f"{row['target_prompt_tokens']:>9,} {row['temperature']:>4.1f} "
              f"{100 * other['acceptance']:>6.1f}% {100 * row['acceptance']:>6.1f}% "
              f"{100 * (row['acceptance'] - other['acceptance']):>+6.1f} "
              f"{other['tokens_per_round']:>7.2f} {row['tokens_per_round']:>7.2f} "
              f"{row['tokens_per_round'] - other['tokens_per_round']:>+7.2f}")
    sa, sb = score_of(a), score_of(b)
    if sa and sb:
        print(f"SCORE  B t/round={sb['tokens_per_round']:.3f} accept={100*sb['acceptance']:.1f}% "
              f"-> A t/round={sa['tokens_per_round']:.3f} accept={100*sa['acceptance']:.1f}%  "
              f"(d t/round {sa['tokens_per_round'] - sb['tokens_per_round']:+.3f}, "
              f"d accept {100*(sa['acceptance'] - sb['acceptance']):+.1f}%)")


def main() -> None:
    global BASE, MODEL
    parser = argparse.ArgumentParser(
        description="DFlash2 fixed acceptance benchmark for before/after comparison")
    parser.add_argument("--base", default=BASE, help="server base URL")
    parser.add_argument("--model", default=MODEL, help="model id as reported by the server")
    parser.add_argument("--quick", action="store_true",
                        help="fast suite: [1024,80000,122000] at T=1")
    parser.add_argument("--temperature", type=float, nargs="+", default=None,
                        help="override suite temperatures")
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=None,
                        help="override suite prompt targets (breaks comparability)")
    parser.add_argument("--draft-tokens", type=int, default=7,
                        help="K the server was started with (must match both runs)")
    parser.add_argument("--no-thinking", action="store_true",
                        help="disable thinking (default is on, matching production)")
    parser.add_argument("--json", metavar="FILE", default=None,
                        help="write the full run (rows + protocol + score) to JSON")
    parser.add_argument("--compare", metavar="FILE", default=None,
                        help="diff this run against a previously saved JSON")
    parser.add_argument("--from-json", metavar="FILE", default=None,
                        help="load rows from a saved JSON and skip generating")
    args = parser.parse_args()

    BASE = args.base
    MODEL = args.model

    if args.from_json:
        with open(args.from_json, encoding="utf-8") as handle:
            results = json.load(handle)["rows"]
        print(f"loaded {len(results)} cells from {args.from_json}")
        score = print_summary(results, f"run from {args.from_json}")
        if args.compare:
            with open(args.compare, encoding="utf-8") as handle:
                compare(results, json.load(handle)["rows"])
        return

    print(f"server: {BASE}  model={MODEL}  K={args.draft_tokens} (server-fixed)")
    print(f"suite : quick={args.quick}  prompts="
          f"{QUICK_PROMPTS if args.quick and not args.prompt_tokens else STANDARD_PROMPTS}"
          + (" (overridden)" if args.prompt_tokens else "") + "\n")

    results = run_suite(args)
    score = print_summary(results, "benchmark")

    record = {
        "protocol": {
            "suite": "quick" if args.quick else "standard",
            "prompts": [r["target_prompt_tokens"] for r in
                        sorted(results, key=lambda x: (x["target_prompt_tokens"],
                                                       -x["temperature"]))],
            "completion": COMPLETION,
            "task": "story",
            "thinking": not args.no_thinking,
            "draft_tokens_K": args.draft_tokens,
            "base": BASE,
            "model": MODEL,
        },
        "score": score,
        "rows": results,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        print(f"\nwrote {args.json}")
    if args.compare:
        with open(args.compare, encoding="utf-8") as handle:
            compare(results, json.load(handle)["rows"])


if __name__ == "__main__":
    main()
