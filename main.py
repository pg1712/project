
import nltk
import torch
import numpy as np
from rank_bm25 import BM25Okapi
from transformers import T5ForConditionalGeneration, T5Tokenizer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


def load_documents(file_path):
    with open(file_path, "r") as f:
        docs = f.read().split("\n")
    return [doc for doc in docs if doc.strip() != ""]

documents = load_documents("data.txt")
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)
print(f"Loaded {len(documents)} documents into BM25 index.\n")

MODEL_NAME = "google/flan-t5-large"
print(f"Loading {MODEL_NAME}...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded.\n")


def format_prompt(docs, query, previous_sentences):
    context = " ".join(docs)
    history = " ".join(previous_sentences)
    if history:
        return (f"Context: {context}\n"
                f"Question: {query}\n"
                f"Answer so far: {history}\n"
                f"Continue the answer in one sentence:")
    return (f"Context: {context}\n"
            f"Question: {query}\n"
            f"Answer in one sentence:")

def generate_with_probs(prompt, max_new_tokens=80):
    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=512, padding=True
    )
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    generated_ids = output.sequences[0][1:]
    token_probs = []
    for i, score in enumerate(output.scores):
        probs = torch.softmax(score[0], dim=-1)
        token_probs.append(probs[generated_ids[i].item()].item())
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text.strip(), token_probs


def generate_with_fallback(prompt, query, token_probs_holder):
    text, probs = generate_with_probs(prompt)
    token_probs_holder.extend(probs)

    if text.lower().strip() in ("unanswerable", "unknown", "i don't know", ""):
        fallback_prompt = f"Answer this question in one sentence: {query}"
        text, probs = generate_with_probs(fallback_prompt)
        token_probs_holder.clear()
        token_probs_holder.extend(probs)

    return text

# ─────────────────────────────────────────
# Retrieve
# ─────────────────────────────────────────
def retrieve(query_text, n=3):
    tokens = query_text.lower().split()
    if not tokens:
        return documents[:n]
    return bm25.get_top_n(tokens, documents, n=n)

# ─────────────────────────────────────────
# Extract First Sentence
# ─────────────────────────────────────────
def extract_first_sentence(text):
    sentences = nltk.sent_tokenize(text)
    return sentences[0].strip() if sentences else text.strip()

# ─────────────────────────────────────────
# Mask Low-Confidence Words
# ─────────────────────────────────────────
def mask_low_confidence_tokens(sentence, token_probs, beta=0.4):
    word_tokens = sentence.split()
    if not word_tokens:
        return sentence
    word_prob_map = []
    subword_idx = 0
    for word in word_tokens:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        n_sub = len(encoded)
        if subword_idx + n_sub <= len(token_probs):
            avg_prob = float(np.mean(token_probs[subword_idx: subword_idx + n_sub]))
        else:
            avg_prob = 1.0
        word_prob_map.append((word, avg_prob))
        subword_idx += n_sub

    kept = [w for w, p in word_prob_map if p >= beta]

    # Safety guard: if masking removed too many words, the query becomes
    # meaningless for BM25. Fall back to top-3 highest-confidence words
    # so retrieval always has something real to work with.
    if len(kept) < 3:
        top3 = sorted(word_prob_map, key=lambda x: x[1], reverse=True)[:3]
        kept = [w for w, _ in top3]

    return " ".join(kept)

# ─────────────────────────────────────────
# FIX 2: Substring-aware deduplication
# "New Delhi" + "New Delhi is the capital..."
# — the second contains the first, so drop
# the shorter one and keep the richer sentence.
# ─────────────────────────────────────────
def deduplicate_sentences(sentences):
    cleaned = [s.lower().strip().rstrip(".") for s in sentences]
    result = []
    for i, s in enumerate(sentences):
        c = cleaned[i]
        # Keep this sentence only if no other sentence already contains it
        dominated = any(
            c != cleaned[j] and c in cleaned[j]
            for j in range(len(sentences)) if j != i
        )
        if not dominated and c not in [cleaned[k] for k in range(i)]:
            result.append(s)
    return result if result else [sentences[-1]]

# ─────────────────────────────────────────
# FIX 3: Smarter completion — stop if the
# current answer already contains the key
# facts (high avg_prob + short answer is fine)
# ─────────────────────────────────────────
def is_complete(sentences, max_sentences=3):
    if len(sentences) >= max_sentences:
        return True
    if len(sentences) >= 2 and sentences[-1] == sentences[-2]:
        return True
    if sentences:
        word_count = sum(len(s.split()) for s in sentences)
        if word_count > 40:
            return True
        # A short confident answer (date, name, place) is already complete
        last = sentences[-1].strip().rstrip(".")
        words = last.split()
        if len(words) <= 6:
            return True
    return False

# ─────────────────────────────────────────
# Exact Match
# ─────────────────────────────────────────
def exact_match(prediction, reference):
    return 1 if reference.lower() in prediction.lower() else 0

# ─────────────────────────────────────────
# BASELINE 1: No Retrieval
# ─────────────────────────────────────────
def no_retrieval(query):
    prompt = f"Answer this question in one sentence: {query}"
    result, _ = generate_with_probs(prompt)
    return result

# ─────────────────────────────────────────
# BASELINE 2: Single-Time RAG
# ─────────────────────────────────────────
def rag(query):
    docs = retrieve(query)
    prompt = format_prompt(docs, query, [])
    probs_holder = []
    result = generate_with_fallback(prompt, query, probs_holder)
    return result

# ─────────────────────────────────────────
# FLARE: Forward-Looking Active Retrieval
# ─────────────────────────────────────────
def flare(query, theta=0.5, beta=0.2, max_steps=5):
    y = []
    retrieval_log = []
    docs = retrieve(query)

    for step in range(max_steps):
        prompt = format_prompt(docs, query, y)
        token_probs = []
        temp_raw = generate_with_fallback(prompt, query, token_probs)
        temp_sentence = extract_first_sentence(temp_raw)

        min_prob = min(token_probs) if token_probs else 1.0
        avg_prob = float(np.mean(token_probs)) if token_probs else 1.0

        if min_prob >= theta:
            y.append(temp_sentence)
            retrieval_log.append({
                "step": step + 1, "triggered": False,
                "min_prob": round(min_prob, 3),
                "avg_prob": round(avg_prob, 3),
                "sentence": temp_sentence,
            })
        else:
            masked_query = mask_low_confidence_tokens(temp_sentence, token_probs, beta)
            docs = retrieve(masked_query)

            new_prompt = format_prompt(docs, query, y)
            new_probs = []
            new_output = generate_with_fallback(new_prompt, query, new_probs)
            final_sentence = extract_first_sentence(new_output)

            y.append(final_sentence)
            retrieval_log.append({
                "step": step + 1, "triggered": True,
                "min_prob": round(min_prob, 3),
                "avg_prob": round(avg_prob, 3),
                "masked_query": masked_query,
                "temp_sentence": temp_sentence,
                "final_sentence": final_sentence,
            })

        if is_complete(y):
            break

    y = deduplicate_sentences(y)
    return " ".join(y), retrieval_log

# ─────────────────────────────────────────
# Print Retrieval Log
# ─────────────────────────────────────────
def print_retrieval_log(log):
    print("\n  [FLARE Retrieval Log]")
    for entry in log:
        s = entry["step"]
        if entry["triggered"]:
            print(f"  Step {s}: RETRIEVAL TRIGGERED "
                  f"(min_prob={entry['min_prob']}, avg_prob={entry['avg_prob']})")
            print(f"    temp      : {entry['temp_sentence']}")
            print(f"    query used: {entry['masked_query']}")
            print(f"    final     : {entry['final_sentence']}")
        else:
            print(f"  Step {s}: accepted "
                  f"(min_prob={entry['min_prob']}, avg_prob={entry['avg_prob']})")
            print(f"    sentence  : {entry['sentence']}")

# ─────────────────────────────────────────
# Test Cases
# ─────────────────────────────────────────
test_cases = [
    {"query": "Who is Joe Biden and where did he study?",
     "reference": "University of Delaware"},
    {"query": "What is machine learning?",
     "reference": "machine learning"},
    {"query": "What is the capital of India?",
     "reference": "New Delhi"},
    {"query": "When did India gain independence?",
     "reference": "1947"},
    {"query": "Who created Python programming language?",
     "reference": "Guido van Rossum"},
    {"query": "What is deep learning?",
     "reference": "neural network"},
]

print("=" * 60)
print(f"FLARE vs RAG vs No Retrieval  [{MODEL_NAME}]")
print("=" * 60)

results = {"no_retrieval": [], "rag": [], "flare": []}

for tc in test_cases:
    q, ref = tc["query"], tc["reference"]
    print(f"\nQuery     : {q}")
    print(f"Reference : '{ref}'")
    print("-" * 55)

    nr      = no_retrieval(q)
    r       = rag(q)
    fl, log = flare(q)

    em_nr = exact_match(nr, ref)
    em_r  = exact_match(r,  ref)
    em_fl = exact_match(fl, ref)

    results["no_retrieval"].append(em_nr)
    results["rag"].append(em_r)
    results["flare"].append(em_fl)

    print(f"No Retrieval [{em_nr}] : {nr[:110]}")
    print(f"RAG          [{em_r}] : {r[:110]}")
    print(f"FLARE        [{em_fl}] : {fl[:110]}")
    print_retrieval_log(log)

print("\n" + "=" * 60)
print("Overall Exact Match Scores")
print("=" * 60)
for method, scores in results.items():
    avg = round(sum(scores) / len(scores), 2)
    bar = "█" * int(avg * 20)
    print(f"  {method:<20} EM = {avg}  {bar}  raw: {scores}")