"""Check that your free Groq key works.
In PowerShell:   $env:GROQ_API_KEY="your_key"     then     python test_llm.py
"""
import os

from openai import OpenAI

key = os.environ.get("GROQ_API_KEY")
if not key:
    raise SystemExit('GROQ_API_KEY is not set. Run  $env:GROQ_API_KEY="your_key"  first.')

client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)
skip = ("whisper", "orpheus", "guard", "safeguard", "tts", "embed", "allam")
models = [m.id for m in client.models.list().data if not any(w in m.id.lower() for w in skip)]
print("Chat models for your key:", models)

for model in models:
    try:
        out = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello in one short sentence."}],
            max_tokens=500,
        )
        print(f"OK    {model}: {(out.choices[0].message.content or '').strip()[:70]}")
    except Exception as e:
        print(f"FAIL  {model}: {type(e).__name__}: {str(e)[:130]}")