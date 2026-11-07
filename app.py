"""Minahil's digital clone: hybrid RAG (BM25 + dense + RRF) with a Gradio UI.

Run the app:               python app.py
Build embeddings once:     python app.py --build
Run the report questions:  python app.py --test
"""
import glob
import json
import os
import re
import sys

import numpy as np
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------- settings
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
EMB_PATH = os.path.join(BASE, "embeddings.npy")
META_PATH = os.path.join(BASE, "chunks.json")

CHUNK_WORDS = 250          # about 200 to 300 words per chunk
OVERLAP_WORDS = 40         # small overlap between neighbouring chunks
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
RRF_K = 60                 # Reciprocal Rank Fusion constant
MODES = ["BM25 only", "Dense only", "Hybrid"]
NOT_KNOWN = "I don't have that in my notes."

PERSONA = """You are the AI clone of Minahil, a 7th-semester BS Computer Science student at UMT Lahore who is interested in AI.
Speak in the first person, in a casual and funny tone. Keep answers short (2 to 5 sentences).
Rules:
1. Answer ONLY using the CONTEXT below. It comes from my own documents.
2. Jokes and a casual tone are fine, but never add facts, times, numbers, names or details that are not written in the CONTEXT.
3. If the context does not contain the answer, say: "I don't have that in my notes." Do not guess or invent facts about me.
4. Never share phone numbers, addresses, ID numbers or other private details.
5. If asked whether you are the real person, say you are an AI clone of Minahil.
CONTEXT:
{context}
QUESTION:
{question}"""

STOPWORDS = set(
    "a an the and or of to in on at for is are was were be been i me my you your we it this "
    "that with as by from do does did what which who how why when where can could would should "
    "about have has had not so if but".split()
)


# ---------------------------------------------------------- load and chunk
def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def load_documents():
    docs = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*"))):
        if path.lower().endswith((".md", ".txt")):
            with open(path, encoding="utf-8") as f:
                docs.append((os.path.basename(path), f.read()))
    return docs


def chunk_text(text, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    """Pack whole paragraphs (an FAQ question and its answer stay together) into
    chunks of about `size` words, repeating up to `overlap` words from the end of
    the previous chunk at the start of the next one."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    chunks, current, current_len = [], [], 0
    for block in blocks:
        n = len(block.split())
        if current and current_len + n > size:
            chunks.append("\n\n".join(current))
            keep, kept = [], 0
            for prev in reversed(current):          # carry over the overlap
                pn = len(prev.split())
                if kept + pn > overlap:
                    break
                keep.insert(0, prev)
                kept += pn
            current, current_len = keep, kept
        current.append(block)
        current_len += n
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def build_chunks():
    chunks = []
    for name, text in load_documents():
        stem = os.path.splitext(name)[0]
        for i, piece in enumerate(chunk_text(text)):
            chunks.append({"source": name, "chunk_id": f"{stem}-{i}", "text": piece})
    return chunks


# --------------------------------------------------------------- retrieval
class Retriever:
    def __init__(self):
        self.chunks = build_chunks()
        if not self.chunks:
            raise RuntimeError("No .md or .txt files found in the data folder.")
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self._model = None
        self.embeddings = self._load_or_build_embeddings()

    def _encode(self, texts):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(EMBED_MODEL)
        vecs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype("float32")

    def build_embeddings(self):
        emb = self._encode([c["text"] for c in self.chunks])
        np.save(EMB_PATH, emb)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=1)
        return emb

    def _load_or_build_embeddings(self):
        if os.path.exists(EMB_PATH) and os.path.exists(META_PATH):
            emb = np.load(EMB_PATH)
            with open(META_PATH, encoding="utf-8") as f:
                saved = json.load(f)
            if saved == self.chunks and emb.shape[0] == len(self.chunks):
                return emb
            print("Saved embeddings do not match the data files, rebuilding...")
        return self.build_embeddings()

    def search(self, question, mode="Hybrid", k=3):
        n = len(self.chunks)
        bm = np.array(self.bm25.get_scores(tokenize(question)), dtype=float)
        de = self.embeddings @ self._encode([question])[0]   # cosine (vectors are normalised)

        bm_order = list(np.argsort(-bm))
        de_order = list(np.argsort(-de))
        bm_rank = {int(i): r + 1 for r, i in enumerate(bm_order)}
        de_rank = {int(i): r + 1 for r, i in enumerate(de_order)}

        if mode == "BM25 only":
            order = [int(i) for i in bm_order if bm[i] > 0][:k]   # no keyword match -> no result
            score = bm
        elif mode == "Dense only":
            order = [int(i) for i in de_order][:k]
            score = de
        else:  # Hybrid: Reciprocal Rank Fusion, score = sum of 1 / (60 + rank)
            rrf = np.zeros(n)
            for i in range(n):
                if bm[i] > 0:                      # chunks with no keyword hit get no BM25 vote
                    rrf[i] += 1.0 / (RRF_K + bm_rank[i])
                rrf[i] += 1.0 / (RRF_K + de_rank[i])
            order = [int(i) for i in np.argsort(-rrf)][:k]
            score = rrf

        results = []
        for i in order:
            c = self.chunks[i]
            results.append({
                "source": c["source"], "chunk_id": c["chunk_id"], "text": c["text"],
                "score": float(score[i]), "bm25_score": float(bm[i]), "dense_score": float(de[i]),
                "bm25_rank": bm_rank[i], "dense_rank": de_rank[i],
            })
        return results


# -------------------------------------------------------------- generation
def templated_answer(results, note=""):
    top = " ".join(results[0]["text"].split())
    snippet = top if len(top) < 450 else top[:450].rsplit(" ", 1)[0] + "..."
    return f"{note}Here is the closest passage from my notes: {snippet}".strip()


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_GROQ_MODEL_CACHE = []


def groq_chat_models(client):
    """Ask Groq which models this key can use and keep the chat ones (model names change over time)."""
    if _GROQ_MODEL_CACHE:
        return _GROQ_MODEL_CACHE
    forced = [os.environ["GROQ_MODEL"]] if os.environ.get("GROQ_MODEL") else []
    try:
        ids = [m.id for m in client.models.list().data]
    except Exception:
        ids = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    skip = ("whisper", "orpheus", "guard", "safeguard", "tts", "embed", "allam")
    ids = [i for i in ids if not any(w in i.lower() for w in skip)]
    prefer = ("gpt-oss-20b", "gpt-oss-120b", "llama", "qwen")
    ids.sort(key=lambda i: next((n for n, p in enumerate(prefer) if p in i.lower()), len(prefer)))
    _GROQ_MODEL_CACHE.extend(forced + ids)
    return _GROQ_MODEL_CACHE


def generate(question, results):
    if not results:
        return NOT_KNOWN
    groq_key = os.environ.get("GROQ_API_KEY")
    hf_token = os.environ.get("HF_TOKEN")
    if not groq_key and not hf_token:
        return templated_answer(results, "(No LLM key set, so this is a templated answer.) ")

    context = "\n\n".join(f"[{r['source']} / {r['chunk_id']}]\n{r['text']}" for r in results)
    messages = [{"role": "user", "content": PERSONA.format(context=context, question=question)}]
    errors = []

    if groq_key:  # free Groq key through its OpenAI-compatible API
        try:
            from openai import OpenAI
            client = OpenAI(base_url=GROQ_BASE_URL, api_key=groq_key, timeout=40)
            for model in groq_chat_models(client)[:3]:
                try:
                    out = client.chat.completions.create(
                        model=model, messages=messages, max_tokens=1000, temperature=0.3)
                    text = (out.choices[0].message.content or "").strip()
                    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
                    if text:
                        return text
                    errors.append(f"{model}: empty answer")
                except Exception as e:
                    errors.append(f"{model}: {type(e).__name__}: {str(e)[:100]}")
        except Exception as e:
            errors.append(f"groq setup: {type(e).__name__}: {str(e)[:100]}")

    if hf_token:  # Hugging Face Inference Providers (needs credits)
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(model=LLM_MODEL, token=hf_token)
            out = client.chat_completion(messages=messages, max_tokens=300, temperature=0.3)
            return out.choices[0].message.content.strip()
        except Exception as e:
            errors.append(f"{LLM_MODEL}: {type(e).__name__}: {str(e)[:100]}")

    return templated_answer(results, f"(LLM call failed: {' | '.join(errors)}) ")


def format_chunks(results, mode):
    if not results:
        return "_No chunks matched this question in this mode._"
    lines = []
    for n, r in enumerate(results, 1):
        if mode == "Hybrid":
            scores = (f"RRF score {r['score']:.4f} | BM25 rank {r['bm25_rank']} "
                      f"(score {r['bm25_score']:.2f}) | Dense rank {r['dense_rank']} "
                      f"(cosine {r['dense_score']:.3f})")
        elif mode == "BM25 only":
            scores = f"BM25 score {r['score']:.2f}"
        else:
            scores = f"Cosine similarity {r['score']:.3f}"
        text = " ".join(r["text"].split())
        text = text if len(text) < 500 else text[:500].rsplit(" ", 1)[0] + "..."
        lines.append(f"**{n}. {r['source']}** (chunk `{r['chunk_id']}`)  \n{scores}\n\n> {text}\n")
    return "\n".join(lines)


# --------------------------------------------------------------------- UI
def build_ui(retriever):
    import gradio as gr

    def ask(question, mode, k):
        question = (question or "").strip()
        if not question:
            return "Please type a question first.", ""
        results = retriever.search(question, mode, int(k))
        return generate(question, results), format_chunks(results, mode)

    with gr.Blocks(title="Minahil's AI Clone") as demo:
        gr.Markdown("# Minahil's Digital Clone\n## ⚠️ This is an AI clone, not the real person.")
        gr.Markdown("Ask me anything about my studies, projects and work. I only answer from my own notes.")
        question = gr.Textbox(label="Your question", placeholder="e.g. What is thecandlescent?")
        with gr.Row():
            mode = gr.Radio(MODES, value="Hybrid", label="Retrieval mode")
            k = gr.Slider(1, 5, value=3, step=1, label="Chunks to retrieve")
        button = gr.Button("Ask", variant="primary")
        answer = gr.Textbox(label="Answer (AI clone)", lines=6)
        gr.Markdown("### Retrieved chunks (source and score)")
        chunks = gr.Markdown()
        button.click(ask, [question, mode, k], [answer, chunks])
        question.submit(ask, [question, mode, k], [answer, chunks])
    return demo


# ------------------------------------------------------------ report tests
TEST_QUESTIONS = [
    "What is HAM10000?",                                  # exact rare keyword
    "Are you a nocturnal person?",                        # paraphrase, no shared words
    "Which algorithm keeps only the best few states at each level?",
    "Which AWS service did you study for object storage?",
    "Do you drink coffee?",
]


def run_tests(retriever):
    rows = ["| Question | BM25 result | Dense result | Hybrid result | Best mode and why |",
            "|---|---|---|---|---|"]
    for i, q in enumerate(TEST_QUESTIONS, 1):
        cells = []
        for mode in MODES:
            res = retriever.search(q, mode, 3)
            cells.append("<br>".join(f"{r['chunk_id']} ({r['score']:.3f})" for r in res) or "no match")
            print(f"\nQ{i}: {q}\n  {mode}:")
            for r in res:
                print(f"    {r['chunk_id']:<22} score={r['score']:.4f}")
        rows.append(f"| Q{i}. {q} | {cells[0]} | {cells[1]} | {cells[2]} | (fill in) |")
    with open(os.path.join(BASE, "test_results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print("\nSaved test_results.md. Paste the table into README.md and fill the last column.")


if __name__ == "__main__":
    retriever = Retriever()
    if "--build" in sys.argv:
        retriever.build_embeddings()
        print(f"Saved {retriever.embeddings.shape} embeddings to {EMB_PATH}")
    elif "--test" in sys.argv:
        run_tests(retriever)
    else:
        build_ui(retriever).launch()