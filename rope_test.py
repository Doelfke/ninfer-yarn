import uuid
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",  # your local server (Ollama/vLLM/llama.cpp)
    api_key="local",  # arbitrary, some servers reject empty
)

def build_prompt(target_tokens: int, needle: str) -> str:
    # ~4 chars/token average; use a realistic corpus if you have one
    chars_per_token = 4
    target_chars = target_tokens * chars_per_token
    unit = "The quick brown fox jumps over the lazy dog. "
    filler = unit * (target_chars // len(unit) + 1)
    body = filler[:target_chars]
    pos = int(len(body) * 0.9)
    return body[:pos] + f" The secret phrase is: {needle}." + body[pos:]

def ask(prompt: str) -> str:
    r = client.chat.completions.create(
        model="qwen3.8-27b",
        messages=[
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": prompt + "\n\nWhat is the secret phrase?"},
        ],
        max_tokens=48,
        temperature=0,
        # qwen3.8-27b is a thinking model; otherwise the thinking prefix alone
        # consumes the whole budget and `content` comes back empty.
        extra_body={"enable_thinking": False},
    )
    content = r.choices[0].message.content
    return (content or "").strip()

# 3 points: well inside, near boundary, well into extended region
for depth in [400_000, 500_000, 600_000, 700_000, 800_000, 900_000, 1_000_000]:
    needle = uuid.uuid4().hex
    answer = ask(build_prompt(depth, needle))
    ok = "✓" if needle in answer else "✗"
    print(f"[{depth:>6,} tok]  {ok}  wanted:{needle}  got:{answer}")