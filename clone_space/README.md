---
title: Minahil's AI Clone
emoji: 🤖
sdk: static
pinned: false
---

# Minahil's Digital Clone (Hybrid RAG)

**Live app (Streamlit): PASTE_YOUR_STREAMLIT_URL_HERE**  
**Source code (GitHub mirror): PASTE_YOUR_GITHUB_URL_HERE**

**This is an AI clone, not the real person.** It answers questions about Minahil in the first person, using only the documents in the `data/` folder. If the answer is not in those documents, it says "I don't have that in my notes."

**How it works:** documents are split into chunks of about 250 words (paragraphs stay together, 40-word overlap, file name and chunk ID kept) → BM25 keyword index (`rank_bm25`) and sentence embeddings (`all-MiniLM-L6-v2`, saved in `embeddings.npy`, cosine similarity with numpy) → the two rankings are merged with Reciprocal Rank Fusion (score = sum of 1 / (60 + rank)) → the top chunks go to a free-tier LLM (Llama on Groq) with a persona prompt → the app shows the answer plus the retrieved chunks, sources and scores. The API key is stored as a Streamlit secret (`GROQ_API_KEY`), never in the code.

## Report

### 1. Knowledge base and why
I used 6 documents written about myself: `bio.md` and `cv.md` (facts), `opinions.md` (habits and opinions), `faq.md` (first-person FAQ with 25 questions), `projects.md` (project descriptions) and `study_notes.md` (long-form study notes). I chose them to cover facts, personality, a long document and an FAQ. For BM25, they contain rare keywords such as HAM10000, MindBridge-RAG, thecandlescent, IDA* and Descript. For dense search, they contain plain descriptive passages, such as "I'm more awake and productive late at night", which can be matched by paraphrased questions. Private details (phone, address, ID numbers, other people's names) are not included.

### 2. Test questions
Q1 is built around an exact rare keyword and Q2 is a paraphrase with no shared words. Fusion method: Reciprocal Rank Fusion. The other common method is a weighted score, `alpha * dense + (1 - alpha) * bm25`, after min-max normalising each score.

| Question | BM25 result | Dense result | Hybrid result | Best mode and why |
|---|---|---|---|---|
| Q1. What is HAM10000? (exact rare keyword) | faq-1 (1.3124)<br>projects-0 (1.2559) | projects-0 (0.1579)<br>faq-1 (0.1363)<br>cv-0 (0.0974) | faq-1 (0.0325)<br>projects-0 (0.0325)<br>cv-0 (0.0159) | **BM25.** It returned only the 2 chunks that contain the exact word "HAM10000". Dense found the same 2 chunks but with weak scores, and Hybrid added an unrelated third chunk (cv-0). |
| Q2. When is your brain sharpest? (paraphrase, no shared words) | no match | study_notes-0 (0.2264)<br>opinions-0 (0.2117)<br>faq-0 (0.1837) | study_notes-0 (0.0164)<br>opinions-0 (0.0161)<br>faq-0 (0.0159) | **Dense.** BM25 found nothing because no words are shared. Dense found the right chunks (opinions-0 and faq-0, which say I am a night owl), but ranked study_notes-0 first by a small margin. Hybrid copied the Dense order because BM25 had no votes. |
| Q3. Which algorithm keeps only the best few states at each level? | study_notes-0 (10.3272)<br>opinions-0 (5.1178)<br>faq-2 (2.1167) | study_notes-0 (0.4040)<br>study_notes-1 (0.1535)<br>faq-2 (0.1512) | study_notes-0 (0.0328)<br>study_notes-1 (0.0318)<br>faq-2 (0.0317) | **Hybrid.** All modes put study_notes-0 (Beam Search) first. BM25's second hit (opinions-0) was unrelated, while Hybrid's top 2 were both study notes chunks (study_notes-0 and study_notes-1). |
| Q4. Which AWS service did you study for object storage? | study_notes-1 (7.6960)<br>cv-0 (1.1737)<br>faq-2 (0.8451) | cv-0 (0.4281)<br>study_notes-1 (0.3854)<br>faq-2 (0.2654) | cv-0 (0.0325)<br>study_notes-1 (0.0325)<br>faq-2 (0.0317) | **BM25.** It ranked study_notes-1 first, the chunk that says S3 is used for object storage. Dense put cv-0 first, which only lists S3. Hybrid tied the two chunks at 0.0325, so cv-0 was listed first. |
| Q5. Do you drink coffee? | opinions-0 (2.8681)<br>faq-1 (0.8723)<br>faq-0 (0.8723) | opinions-0 (0.1804)<br>faq-1 (0.1222)<br>faq-3 (0.0723) | opinions-0 (0.0328)<br>faq-1 (0.0323)<br>faq-0 (0.0315) | **Tie.** All 3 modes put opinions-0 first, and it contains the answer ("neither coffee nor tea"). |

**Why the modes differ.** BM25 matches exact words, so it wins on rare keywords (Q1, Q4) but returns nothing when the question uses different words (Q2). Dense search matches meaning, so it handles paraphrases, but its cosine scores are low and it can rank a loosely related chunk first (Q2, Q4). The raw scores are on different scales (BM25 up to about 10, cosine about 0.0 to 0.4, RRF about 0.03), so Hybrid fuses ranks instead of scores. Hybrid changed the ranking compared with BM25 in Q1, Q3 and Q4.

### 3. Failure case
### 3. Failure case
**Question:** "When is your brain sharpest?" (mode: Dense only, 1 chunk retrieved)

**What happened:** The clone answered "I don't have that in my notes." even though my notes say I am a night owl and more productive late at night (in `opinions.md` and `faq.md`).

**What it retrieved:** Only `study_notes-0` (cosine similarity 0.22), which is about AI search algorithms, compilers and cloud services. The correct chunk, `opinions-0`, scored 0.21 and ranked second, so it was cut off when only one chunk was kept.

**Why it happened:** The two scores were almost tied (0.2264 against 0.2117), so a tiny difference decided the result, and retrieving just one chunk made the miss fatal. The question is short and abstract and has no keywords, so BM25 returned nothing and dense search had to guess from meaning alone. It most likely matched the general "studying and exam preparation" theme of `study_notes-0`. Also, `opinions-0` mixes six topics (work habits, views on AI, advice, plans, preferences) in one chunk, so its embedding is a blend and the night-owl sentence has little weight. The clone behaved honestly, because it refused to guess, but the retrieval step missed the evidence.
**Follow-up:** With 3 chunks and Dense only, the clone answered correctly (a night owl, most productive late at night), so the failure above came from keeping only 1 chunk. In that run, however, the answer also added "if you need me at 3 am", which is not in my notes. I fixed this by adding a rule to the prompt that jokes must not add facts, times or numbers.

### 4. One improvement
In Q1 and Q4, two chunks received identical Reciprocal Rank Fusion scores (0.0325), because each ranked first in one method and second in the other, so the order came from their position in the file. I would break ties using the dense similarity or the normalised BM25 score, and add a cross-encoder reranker over the top results so the best chunk is ranked first more reliably.