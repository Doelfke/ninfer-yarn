"""
YaRN needle-probe for qwen3.8-27b over the local OpenAI-compatible server.

The previous version guessed prompt size with `4 chars/token`. Qwen's tokenizer
does NOT compress that repetitive filler at 4 chars/token, so a "400,000-token"
prompt only tokenized to ~355,621 real tokens and a "500,000" one tokenized to
>400,000 real tokens (-> HTTP 400 context length exceeded). That gap is what made
the ceiling look like ~355k when the real limit is `--max-context` (advertised by
GET /v1/models -> max_model_len).

This version measures REAL token counts with the server's own tokenizer
(POST /v1/messages/count_tokens) and sizes the prompt to a target real-token
count that stays below the advertised ceiling.
"""

import json
import time
import uuid
import urllib.request

from openai import OpenAI

BASE = "http://localhost:1234"
MODEL = "qwen3.8-27b"          # matches the model id the server reports
API_KEY = "local"             # arbitrary when the server has no --api-key set
SYSTEM = "Answer concisely."
QUESTION = "\n\nWhat is the secret phrase?"
UNIT = "The quick brown fox jumps over the lazy dog. "   # 45 chars
SAFETY_MARGIN = 128           # room for the system message + question + template


def http_json(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def max_model_len() -> int:
    """Real per-request ceiling the server configured (--max-context)."""
    return int(http_json("GET", "/v1/models")["data"][0]["max_model_len"])


def count_tokens(user_text: str) -> int:
    """Exact prompt token count, as the engine's tokenizer will see it."""
    body = {"model": MODEL, "system": SYSTEM,
            "messages": [{"role": "user", "content": user_text}]}
    return int(http_json("POST", "/v1/messages/count_tokens", body)["input_tokens"])


def build_prompt(filler_chars: int, needle: str) -> str:
    body = (UNIT * (filler_chars // len(UNIT) + 1))[:filler_chars]
    pos = int(len(body) * 0.9)
    return body[:pos] + f" The secret phrase is: {needle}." + body[pos:] + QUESTION


def fit_prompt(target_tokens: int) -> tuple[int, int]:
    """Binary-search filler length so the REAL prompt is ~= target_tokens."""
    dummy = "0" * 32  # same length as uuid.hex -> identical token count
    lo, hi = 0, target_tokens * 5
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(build_prompt(mid, dummy)) <= target_tokens:
            lo = mid
        else:
            hi = mid - 1
    return lo, count_tokens(build_prompt(lo, dummy))


client = OpenAI(base_url=BASE + "/v1", api_key=API_KEY)


def ask(prompt: str):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_tokens=48,
        temperature=0,
        # thinking model: otherwise the thinking prefix alone burns the budget
        # and `content` comes back empty.
        extra_body={"enable_thinking": False},
    )
    content = (r.choices[0].message.content or "").strip()
    actual = r.usage.prompt_tokens if r.usage else -1
    return content, actual


def main():
    ceiling = max_model_len()
    print(f"server ceiling (GET /v1/models max_model_len) = {ceiling:,} tokens\n")

    # real-token depths: well inside, near, and right against the ceiling
    targets = [
        int(ceiling * 0.50),
        int(ceiling * 0.90),
        ceiling - SAFETY_MARGIN - 1,   # just under the hard limit
    ]

    for target in targets:
        needle = uuid.uuid4().hex
        filler, measured = fit_prompt(target)   # size to REAL tokens
        prompt = build_prompt(filler, needle)
        t0 = time.time()
        try:
            answer, actual = ask(prompt)
        except Exception as exc:   # e.g. openai.BadRequestError (HTTP 400)
            print(f"[target {target:>7,}] fit {measured:>7,} real tok -> ERROR: {exc}")
            continue
        ok = "✓" if needle in answer else "✗"
        print(f"[target {target:>7,}] fit {measured:>7,} real tok"
              f" | prompt {actual:>7,} | {time.time()-t0:7.1f}s | {ok} got:{answer!r}")


if __name__ == "__main__":
    main()