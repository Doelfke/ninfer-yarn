import uuid
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",  # your local server (Ollama/vLLM/llama.cpp)
    api_key="local",  # arbitrary, some servers reject empty
)

def build_prompt(target_tokens: int, needle: str) -> str:
    # ~4 chars/token average; use a realistic corpus if you have one
    filler = "The quick brown fox jumps over the lazy dog. " * (target_tokens // 20 * 3)
    body = filler[:target_tokens]
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
for depth in [50000, 100_000, 200_000, 262_000, 380_000]:
    needle = uuid.uuid4().hex
    answer = ask(build_prompt(depth, needle))
    ok = "✓" if needle in answer else "✗"
    print(f"[{depth:>6,} tok]  {ok}  wanted:{needle}  got:{answer}")