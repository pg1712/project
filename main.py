# # # # import nltk
# # # # from rank_bm25 import BM25Okapi
# # # # from transformers import pipeline

# # # # nltk.download('punkt')

# # # # # -----------------------------
# # # # # Load Dataset
# # # # # -----------------------------
# # # # def load_documents(file_path):
# # # #     with open(file_path, "r") as f:
# # # #         docs = f.read().split("\n")
# # # #     return [doc for doc in docs if doc.strip() != ""]

# # # # documents = load_documents("data.txt")

# # # # # Tokenize for BM25
# # # # tokenized_docs = [doc.lower().split() for doc in documents]
# # # # bm25 = BM25Okapi(tokenized_docs)

# # # # # -----------------------------
# # # # # Load FREE LLM (Local)
# # # # # -----------------------------
# # # # generator = pipeline("text-generation", model="gpt2-medium")

# # # # # -----------------------------
# # # # # Helper: Generate text
# # # # # -----------------------------
# # # # # def generate_text(prompt, max_length=100):
# # # # #     result = generator(prompt, max_length=max_length, num_return_sequences=1)
# # # # #     return result[0]["generated_text"]
# # # # def generate_text(prompt):
# # # #     prompt = "Answer the question clearly: " + prompt
    
# # # #     result = generator(
# # # #         prompt,
# # # #         max_new_tokens=50,
# # # #         do_sample=True,
# # # #         temperature=0.7,
# # # #         pad_token_id=50256
# # # #     )
    
# # # #     return result[0]["generated_text"]
# # # # # -----------------------------
# # # # # BASELINE 1: No Retrieval
# # # # # -----------------------------
# # # # def no_retrieval(query):
# # # #     return generate_text(query)

# # # # # -----------------------------
# # # # # BASELINE 2: RAG (Single Retrieval)
# # # # # -----------------------------
# # # # def rag(query):
# # # #     tokenized_query = query.lower().split()
# # # #     top_docs = bm25.get_top_n(tokenized_query, documents, n=2)
    
# # # #     context = " ".join(top_docs)
# # # #     prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    
# # # #     return generate_text(prompt)

# # # # # -----------------------------
# # # # # SIMPLE CONFIDENCE CHECK
# # # # # -----------------------------
# # # # def low_confidence(sentence):
# # # #     # Simple heuristic instead of token probabilities
# # # #     uncertain_words = ["maybe", "might", "possibly", "unknown"]
# # # #     return any(word in sentence.lower() for word in uncertain_words)

# # # # # -----------------------------
# # # # # FLARE (Simplified)
# # # # # -----------------------------
# # # # # def flare(query):
# # # # #     answer = "Answer the question: " + query

# # # # #     for step in range(1):  # limit steps
# # # # #         sentence = generate_text(answer)

# # # # #         if low_confidence(sentence):
# # # # #             tokenized_query = sentence.lower().split()
# # # # #             top_docs = bm25.get_top_n(tokenized_query, documents, n=2)
            
# # # # #             context = " ".join(top_docs)
# # # # #             sentence = generate_text(context + " " + answer)

# # # # #         answer += " " + sentence

# # # # #     return answer
# # # # def flare(query, theta=0.5, max_steps=5):
# # # #     y = []
# # # #     docs = retrieve(query)  # initial retrieval
    
# # # #     for step in range(max_steps):
# # # #         # Generate temp sentence WITH token probabilities
# # # #         temp_sentence, token_probs = generate_with_probs(
# # # #             prompt=format_prompt(docs, query, y)
# # # #         )
        
# # # #         # Check confidence
# # # #         if min(token_probs) >= theta:
# # # #             y.append(temp_sentence)   # accept as-is
# # # #         else:
# # # #             # Form query by masking low-prob tokens
# # # #             masked_query = mask_low_confidence(temp_sentence, token_probs, beta=0.4)
# # # #             docs = retrieve(masked_query)   # new retrieval
# # # #             # Regenerate conditioned on new docs
# # # #             final_sentence, _ = generate_with_probs(
# # # #                 prompt=format_prompt(docs, query, y)
# # # #             )
# # # #             y.append(final_sentence)
        
# # # #         if is_complete(y):
# # # #             break
    
# # # #     return " ".join(y)
# # # # # -----------------------------
# # # # # TEST QUERY
# # # # # -----------------------------
# # # # query = "Who is Joe Biden and where did he study?"

# # # # print("\n--- No Retrieval ---")
# # # # print(no_retrieval(query))

# # # # print("\n--- RAG ---")
# # # # print(rag(query))

# # # # print("\n--- FLARE ---")
# # # # print(flare(query))

# # # import nltk
# # # import torch
# # # import numpy as np
# # # from rank_bm25 import BM25Okapi
# # # from transformers import GPT2LMHeadModel, GPT2Tokenizer

# # # nltk.download('punkt', quiet=True)
# # # nltk.download('punkt_tab', quiet=True)

# # # # ─────────────────────────────────────────
# # # # Load Dataset
# # # # ─────────────────────────────────────────
# # # def load_documents(file_path):
# # #     with open(file_path, "r") as f:
# # #         docs = f.read().split("\n")
# # #     return [doc for doc in docs if doc.strip() != ""]

# # # documents = load_documents("data.txt")
# # # tokenized_docs = [doc.lower().split() for doc in documents]
# # # bm25 = BM25Okapi(tokenized_docs)

# # # # ─────────────────────────────────────────
# # # # Load Model (GPT-2 Medium)
# # # # ─────────────────────────────────────────
# # # print("Loading model...")
# # # model_name = "gpt2-medium"
# # # tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# # # model = GPT2LMHeadModel.from_pretrained(model_name)
# # # model.eval()
# # # tokenizer.pad_token = tokenizer.eos_token
# # # print("Model loaded.\n")

# # # # ─────────────────────────────────────────
# # # # Core: Generate with Token Probabilities
# # # # ─────────────────────────────────────────
# # # def generate_with_probs(prompt, max_new_tokens=60):
# # #     """
# # #     Returns (generated_text, list_of_token_probs).
# # #     token_probs[i] is the probability the model assigned
# # #     to the i-th generated token at the moment it was sampled.
# # #     This is the key mechanism FLARE uses for confidence detection.
# # #     """
# # #     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=900)
# # #     input_ids = inputs["input_ids"]

# # #     with torch.no_grad():
# # #         output = model.generate(
# # #             input_ids,
# # #             max_new_tokens=max_new_tokens,
# # #             do_sample=False,          # greedy — deterministic, more calibrated probs
# # #             temperature=1.0,
# # #             pad_token_id=tokenizer.eos_token_id,
# # #             return_dict_in_generate=True,
# # #             output_scores=True,       # <-- gives us per-step logits
# # #         )

# # #     # output.scores: tuple of (vocab_size,) tensors, one per generated token
# # #     generated_ids = output.sequences[0][input_ids.shape[1]:]  # only new tokens
# # #     token_probs = []
# # #     for i, score in enumerate(output.scores):
# # #         probs = torch.softmax(score[0], dim=-1)
# # #         token_id = generated_ids[i].item()
# # #         token_probs.append(probs[token_id].item())

# # #     generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
# # #     return generated_text.strip(), token_probs


# # # # ─────────────────────────────────────────
# # # # Helper: Retrieve Documents
# # # # ─────────────────────────────────────────
# # # def retrieve(query_text, n=2):
# # #     tokens = query_text.lower().split()
# # #     return bm25.get_top_n(tokens, documents, n=n)


# # # # ─────────────────────────────────────────
# # # # Helper: Format Prompt
# # # # ─────────────────────────────────────────
# # # def format_prompt(docs, query, previous_sentences):
# # #     context = " ".join(docs)
# # #     history = " ".join(previous_sentences)
# # #     if history:
# # #         return f"Context: {context}\nQuestion: {query}\nAnswer so far: {history}\nContinue:"
# # #     return f"Context: {context}\nQuestion: {query}\nAnswer:"


# # # # ─────────────────────────────────────────
# # # # Helper: Mask Low-Confidence Tokens
# # # # ─────────────────────────────────────────
# # # def mask_low_confidence_tokens(sentence, token_probs, beta=0.4):
# # #     """
# # #     Replaces tokens whose probability < beta with a blank.
# # #     This is the 'implicit query' method from the paper (Section 3.2.2).
# # #     Removing uncertain tokens prevents hallucinated values from
# # #     misleading BM25 retrieval.
# # #     """
# # #     words = tokenizer.tokenize(sentence)
# # #     # Align word-level tokens with probability list (lengths may differ slightly)
# # #     n = min(len(words), len(token_probs))
# # #     masked = []
# # #     for i in range(n):
# # #         if token_probs[i] >= beta:
# # #             masked.append(tokenizer.convert_tokens_to_string([words[i]]))
# # #     # Fall back to full sentence if everything was masked
# # #     return " ".join(masked).strip() if masked else sentence


# # # # ─────────────────────────────────────────
# # # # Helper: Extract First Sentence
# # # # ─────────────────────────────────────────
# # # def extract_first_sentence(text):
# # #     """
# # #     The paper uses sentence boundaries as the iteration unit —
# # #     sentences are 'significant semantic units, neither too short
# # #     nor too lengthy' (Section 3.2.1).
# # #     """
# # #     sentences = nltk.sent_tokenize(text)
# # #     return sentences[0] if sentences else text


# # # # ─────────────────────────────────────────
# # # # Helper: Is Answer Complete?
# # # # ─────────────────────────────────────────
# # # def is_complete(sentences, max_sentences=4):
# # #     if len(sentences) >= max_sentences:
# # #         return True
# # #     if sentences:
# # #         last = sentences[-1].strip()
# # #         if last.endswith(".") or last.endswith("?") or last.endswith("!"):
# # #             # Rough heuristic: stop after a complete-looking final sentence
# # #             word_count = sum(len(s.split()) for s in sentences)
# # #             if word_count > 40:
# # #                 return True
# # #     return False


# # # # ─────────────────────────────────────────
# # # # BASELINE 1: No Retrieval
# # # # ─────────────────────────────────────────
# # # def no_retrieval(query):
# # #     prompt = f"Question: {query}\nAnswer:"
# # #     result, _ = generate_with_probs(prompt)
# # #     return result


# # # # ─────────────────────────────────────────
# # # # BASELINE 2: Single-Time RAG
# # # # ─────────────────────────────────────────
# # # def rag(query):
# # #     docs = retrieve(query)
# # #     prompt = format_prompt(docs, query, [])
# # #     result, _ = generate_with_probs(prompt)
# # #     return result


# # # # ─────────────────────────────────────────
# # # # FLARE: Forward-Looking Active Retrieval
# # # # ─────────────────────────────────────────
# # # def flare(query, theta=0.5, beta=0.4, max_steps=5):
# # #     """
# # #     Implementation of FLAREدirect (Section 3.2).

# # #     theta : confidence threshold — any token prob below this triggers retrieval
# # #     beta  : masking threshold   — tokens below this are masked in the query
    
# # #     Key difference from plain RAG:
# # #       - RAG retrieves ONCE based on the input.
# # #       - FLARE retrieves ITERATIVELY based on what it's *about to generate*,
# # #         and only when it's uncertain — avoiding unnecessary retrieval noise.
# # #     """
# # #     y = []                          # accepted sentences so far
# # #     retrieval_log = []              # track when/why retrieval was triggered
# # #     docs = retrieve(query)          # initial retrieval using the input

# # #     for step in range(max_steps):
# # #         prompt = format_prompt(docs, query, y)

# # #         # Step 1: Generate a temporary next sentence with token probabilities
# # #         raw_output, token_probs = generate_with_probs(prompt)
# # #         temp_sentence = extract_first_sentence(raw_output)

# # #         min_prob = min(token_probs) if token_probs else 1.0
# # #         avg_prob = np.mean(token_probs) if token_probs else 1.0

# # #         if min_prob >= theta:
# # #             # Model is confident — accept the sentence without new retrieval
# # #             y.append(temp_sentence)
# # #             retrieval_log.append({
# # #                 "step": step + 1,
# # #                 "triggered": False,
# # #                 "min_prob": round(min_prob, 3),
# # #                 "avg_prob": round(avg_prob, 3),
# # #                 "sentence": temp_sentence,
# # #             })
# # #         else:
# # #             # Model is uncertain — form a new query from the temp sentence
# # #             # by masking its low-confidence tokens (implicit query method)
# # #             masked_query = mask_low_confidence_tokens(temp_sentence, token_probs, beta)
# # #             docs = retrieve(masked_query)   # retrieve based on FUTURE intent

# # #             # Regenerate the sentence conditioned on newly retrieved docs
# # #             new_prompt = format_prompt(docs, query, y)
# # #             new_output, new_probs = generate_with_probs(new_prompt)
# # #             final_sentence = extract_first_sentence(new_output)

# # #             y.append(final_sentence)
# # #             retrieval_log.append({
# # #                 "step": step + 1,
# # #                 "triggered": True,
# # #                 "min_prob": round(min_prob, 3),
# # #                 "avg_prob": round(avg_prob, 3),
# # #                 "masked_query": masked_query,
# # #                 "temp_sentence": temp_sentence,
# # #                 "final_sentence": final_sentence,
# # #             })

# # #         if is_complete(y):
# # #             break

# # #     return " ".join(y), retrieval_log


# # # # ─────────────────────────────────────────
# # # # Display Retrieval Log (for demo/review)
# # # # ─────────────────────────────────────────
# # # def print_retrieval_log(log):
# # #     print("\n  [FLARE Retrieval Log]")
# # #     for entry in log:
# # #         step = entry["step"]
# # #         if entry["triggered"]:
# # #             print(f"  Step {step}: RETRIEVAL TRIGGERED")
# # #             print(f"    min_prob={entry['min_prob']}  avg_prob={entry['avg_prob']}")
# # #             print(f"    temp sentence : {entry['temp_sentence']}")
# # #             print(f"    masked query  : {entry['masked_query']}")
# # #             print(f"    final sentence: {entry['final_sentence']}")
# # #         else:
# # #             print(f"  Step {step}: accepted (min_prob={entry['min_prob']}, avg_prob={entry['avg_prob']})")
# # #             print(f"    sentence: {entry['sentence']}")


# # # # ─────────────────────────────────────────
# # # # Evaluation: Exact Match
# # # # ─────────────────────────────────────────
# # # def exact_match(prediction, reference):
# # #     pred = prediction.lower().strip().rstrip(".")
# # #     ref  = reference.lower().strip().rstrip(".")
# # #     return 1 if ref in pred else 0


# # # # ─────────────────────────────────────────
# # # # Test Queries with Reference Answers
# # # # ─────────────────────────────────────────
# # # test_cases = [
# # #     {
# # #         "query": "Who is Joe Biden and where did he study?",
# # #         "reference": "University of Delaware"
# # #     },
# # #     {
# # #         "query": "What is machine learning?",
# # #         "reference": "machine learning"
# # #     },
# # #     {
# # #         "query": "What is the capital of India?",
# # #         "reference": "New Delhi"
# # #     },
# # # ]

# # # print("=" * 60)
# # # print("FLARE vs RAG vs No Retrieval — Comparison")
# # # print("=" * 60)

# # # results = {"no_retrieval": [], "rag": [], "flare": []}

# # # for tc in test_cases:
# # #     q   = tc["query"]
# # #     ref = tc["reference"]
# # #     print(f"\nQuery: {q}")
# # #     print(f"Reference answer contains: '{ref}'")
# # #     print("-" * 50)

# # #     nr  = no_retrieval(q)
# # #     r   = rag(q)
# # #     fl, log = flare(q)

# # #     em_nr = exact_match(nr,  ref)
# # #     em_r  = exact_match(r,   ref)
# # #     em_fl = exact_match(fl,  ref)

# # #     results["no_retrieval"].append(em_nr)
# # #     results["rag"].append(em_r)
# # #     results["flare"].append(em_fl)

# # #     print(f"No Retrieval : {nr[:120]}")
# # #     print(f"  EM={em_nr}")
# # #     print(f"RAG          : {r[:120]}")
# # #     print(f"  EM={em_r}")
# # #     print(f"FLARE        : {fl[:120]}")
# # #     print(f"  EM={em_fl}")
# # #     print_retrieval_log(log)

# # # print("\n" + "=" * 60)
# # # print("Overall Exact Match Scores")
# # # print("=" * 60)
# # # for method, scores in results.items():
# # #     avg = round(sum(scores) / len(scores), 2)
# # #     print(f"  {method:<20} EM = {avg}  (raw: {scores})")

# # import nltk
# # import torch
# # import numpy as np
# # from rank_bm25 import BM25Okapi
# # from transformers import T5ForConditionalGeneration, T5Tokenizer

# # nltk.download('punkt', quiet=True)
# # nltk.download('punkt_tab', quiet=True)

# # # ─────────────────────────────────────────
# # # Load Dataset
# # # ─────────────────────────────────────────
# # def load_documents(file_path):
# #     with open(file_path, "r") as f:
# #         docs = f.read().split("\n")
# #     return [doc for doc in docs if doc.strip() != ""]

# # documents = load_documents("data.txt")
# # tokenized_docs = [doc.lower().split() for doc in documents]
# # bm25 = BM25Okapi(tokenized_docs)

# # # ─────────────────────────────────────────
# # # Model choice — swap this one line to upgrade:
# # #   "google/flan-t5-base"   ~1GB RAM  (CPU safe)
# # #   "google/flan-t5-large"  ~3GB RAM  (CPU, best free option)
# # #   "google/flan-t5-xl"     ~8GB RAM  (needs GPU)
# # # ─────────────────────────────────────────
# # MODEL_NAME = "google/flan-t5-large"

# # print(f"Loading {MODEL_NAME}...")
# # tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
# # model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
# # model.eval()
# # print("Model loaded.\n")

# # # ─────────────────────────────────────────
# # # Prompt Format
# # # Flan-T5 is instruction-tuned, so we CAN and
# # # SHOULD use explicit instruction prompts — unlike
# # # GPT-2 which would just copy the template.
# # # ─────────────────────────────────────────
# # def format_prompt(docs, query, previous_sentences):
# #     context = " ".join(docs)
# #     history = " ".join(previous_sentences)
# #     if history:
# #         return (f"Context: {context}\n"
# #                 f"Question: {query}\n"
# #                 f"Answer so far: {history}\n"
# #                 f"Continue the answer in one sentence:")
# #     return (f"Context: {context}\n"
# #             f"Question: {query}\n"
# #             f"Answer in one sentence:")

# # # ─────────────────────────────────────────
# # # Core: Generate with Token Probabilities
# # #
# # # Flan-T5 is an encoder-decoder (seq2seq) model,
# # # so generation works differently from GPT-2:
# # #   - Input is encoded by the encoder
# # #   - Decoder generates tokens one by one
# # #   - output.scores gives decoder token logits
# # #   - We use beam_size=1 (greedy) for calibrated probs
# # # ─────────────────────────────────────────
# # def generate_with_probs(prompt, max_new_tokens=80):
# #     inputs = tokenizer(
# #         prompt,
# #         return_tensors="pt",
# #         truncation=True,
# #         max_length=512,
# #         padding=True
# #     )

# #     with torch.no_grad():
# #         output = model.generate(
# #             input_ids=inputs["input_ids"],
# #             attention_mask=inputs["attention_mask"],
# #             max_new_tokens=max_new_tokens,
# #             do_sample=False,            # greedy = calibrated probs
# #             return_dict_in_generate=True,
# #             output_scores=True,
# #         )

# #     # output.sequences[0] includes the decoder start token at position 0
# #     # Skip it: generated token ids start at index 1
# #     generated_ids = output.sequences[0][1:]

# #     token_probs = []
# #     for i, score in enumerate(output.scores):
# #         probs = torch.softmax(score[0], dim=-1)
# #         token_id = generated_ids[i].item()
# #         token_probs.append(probs[token_id].item())

# #     generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
# #     return generated_text.strip(), token_probs

# # # ─────────────────────────────────────────
# # # Retrieve Documents
# # # ─────────────────────────────────────────
# # def retrieve(query_text, n=2):
# #     tokens = query_text.lower().split()
# #     if not tokens:
# #         return documents[:n]
# #     return bm25.get_top_n(tokens, documents, n=n)

# # # ─────────────────────────────────────────
# # # Extract First Sentence
# # # ─────────────────────────────────────────
# # def extract_first_sentence(text):
# #     sentences = nltk.sent_tokenize(text)
# #     return sentences[0].strip() if sentences else text.strip()

# # # ─────────────────────────────────────────
# # # Mask Low-Confidence Words (word-level)
# # # ─────────────────────────────────────────
# # def mask_low_confidence_tokens(sentence, token_probs, beta=0.4):
# #     word_tokens = sentence.split()
# #     if not word_tokens:
# #         return sentence

# #     word_prob_map = []
# #     subword_idx = 0
# #     for word in word_tokens:
# #         encoded = tokenizer.encode(word, add_special_tokens=False)
# #         n_sub = len(encoded)
# #         if subword_idx + n_sub <= len(token_probs):
# #             avg_prob = float(np.mean(token_probs[subword_idx: subword_idx + n_sub]))
# #         else:
# #             avg_prob = 1.0
# #         word_prob_map.append((word, avg_prob))
# #         subword_idx += n_sub

# #     kept = [w for w, p in word_prob_map if p >= beta]
# #     return " ".join(kept) if kept else sentence

# # # ─────────────────────────────────────────
# # # Completion Check
# # # ─────────────────────────────────────────
# # def is_complete(sentences, max_sentences=4):
# #     if len(sentences) >= max_sentences:
# #         return True
# #     if len(sentences) >= 2 and sentences[-1] == sentences[-2]:
# #         return True
# #     if sentences:
# #         word_count = sum(len(s.split()) for s in sentences)
# #         if word_count > 60:
# #             return True
# #     return False

# # # ─────────────────────────────────────────
# # # Exact Match
# # # ─────────────────────────────────────────
# # def exact_match(prediction, reference):
# #     return 1 if reference.lower() in prediction.lower() else 0

# # # ─────────────────────────────────────────
# # # BASELINE 1: No Retrieval
# # # ─────────────────────────────────────────
# # def no_retrieval(query):
# #     prompt = f"Answer this question in one sentence: {query}"
# #     result, _ = generate_with_probs(prompt)
# #     return result

# # # ─────────────────────────────────────────
# # # BASELINE 2: Single-Time RAG
# # # ─────────────────────────────────────────
# # def rag(query):
# #     docs = retrieve(query)
# #     prompt = format_prompt(docs, query, [])
# #     result, _ = generate_with_probs(prompt)
# #     return result

# # # ─────────────────────────────────────────
# # # FLARE: Forward-Looking Active Retrieval
# # # ─────────────────────────────────────────
# # def flare(query, theta=0.5, beta=0.4, max_steps=5):
# #     y = []
# #     retrieval_log = []
# #     docs = retrieve(query)

# #     for step in range(max_steps):
# #         prompt = format_prompt(docs, query, y)
# #         temp_raw, token_probs = generate_with_probs(prompt)
# #         temp_sentence = extract_first_sentence(temp_raw)

# #         min_prob = min(token_probs) if token_probs else 1.0
# #         avg_prob = float(np.mean(token_probs)) if token_probs else 1.0

# #         if min_prob >= theta:
# #             y.append(temp_sentence)
# #             retrieval_log.append({
# #                 "step": step + 1,
# #                 "triggered": False,
# #                 "min_prob": round(min_prob, 3),
# #                 "avg_prob": round(avg_prob, 3),
# #                 "sentence": temp_sentence,
# #             })
# #         else:
# #             masked_query = mask_low_confidence_tokens(temp_sentence, token_probs, beta)
# #             docs = retrieve(masked_query)

# #             new_prompt = format_prompt(docs, query, y)
# #             new_output, _ = generate_with_probs(new_prompt)
# #             final_sentence = extract_first_sentence(new_output)

# #             y.append(final_sentence)
# #             retrieval_log.append({
# #                 "step": step + 1,
# #                 "triggered": True,
# #                 "min_prob": round(min_prob, 3),
# #                 "avg_prob": round(avg_prob, 3),
# #                 "masked_query": masked_query,
# #                 "temp_sentence": temp_sentence,
# #                 "final_sentence": final_sentence,
# #             })

# #         if is_complete(y):
# #             break

# #     return " ".join(y), retrieval_log

# # # ─────────────────────────────────────────
# # # Print Retrieval Log
# # # ─────────────────────────────────────────
# # def print_retrieval_log(log):
# #     print("\n  [FLARE Retrieval Log]")
# #     for entry in log:
# #         s = entry["step"]
# #         if entry["triggered"]:
# #             print(f"  Step {s}: RETRIEVAL TRIGGERED  "
# #                   f"(min_prob={entry['min_prob']}, avg_prob={entry['avg_prob']})")
# #             print(f"    temp      : {entry['temp_sentence']}")
# #             print(f"    query used: {entry['masked_query']}")
# #             print(f"    final     : {entry['final_sentence']}")
# #         else:
# #             print(f"  Step {s}: accepted  "
# #                   f"(min_prob={entry['min_prob']}, avg_prob={entry['avg_prob']})")
# #             print(f"    sentence  : {entry['sentence']}")

# # # ─────────────────────────────────────────
# # # Test Cases
# # # ─────────────────────────────────────────
# # test_cases = [
# #     {"query": "Who is Joe Biden and where did he study?",
# #      "reference": "University of Delaware"},
# #     {"query": "What is machine learning?",
# #      "reference": "machine learning"},
# #     {"query": "What is the capital of India?",
# #      "reference": "New Delhi"},
# # ]

# # print("=" * 60)
# # print(f"FLARE vs RAG vs No Retrieval  [{MODEL_NAME}]")
# # print("=" * 60)

# # results = {"no_retrieval": [], "rag": [], "flare": []}

# # for tc in test_cases:
# #     q, ref = tc["query"], tc["reference"]
# #     print(f"\nQuery     : {q}")
# #     print(f"Reference : '{ref}'")
# #     print("-" * 55)

# #     nr      = no_retrieval(q)
# #     r       = rag(q)
# #     fl, log = flare(q)

# #     em_nr = exact_match(nr, ref)
# #     em_r  = exact_match(r,  ref)
# #     em_fl = exact_match(fl, ref)

# #     results["no_retrieval"].append(em_nr)
# #     results["rag"].append(em_r)
# #     results["flare"].append(em_fl)

# #     print(f"No Retrieval [{em_nr}] : {nr[:110]}")
# #     print(f"RAG          [{em_r}] : {r[:110]}")
# #     print(f"FLARE        [{em_fl}] : {fl[:110]}")
# #     print_retrieval_log(log)

# # print("\n" + "=" * 60)
# # print("Overall Exact Match Scores")
# # print("=" * 60)
# # for method, scores in results.items():
# #     avg = round(sum(scores) / len(scores), 2)
# #     print(f"  {method:<20} EM = {avg}   raw: {scores}")

# import nltk
# import torch
# import numpy as np
# from rank_bm25 import BM25Okapi
# from transformers import T5ForConditionalGeneration, T5Tokenizer

# nltk.download('punkt', quiet=True)
# nltk.download('punkt_tab', quiet=True)

# # ─────────────────────────────────────────
# # Load Dataset
# # ─────────────────────────────────────────
# def load_documents(file_path):
#     with open(file_path, "r") as f:
#         docs = f.read().split("\n")
#     return [doc for doc in docs if doc.strip() != ""]

# documents = load_documents("data.txt")
# tokenized_docs = [doc.lower().split() for doc in documents]
# bm25 = BM25Okapi(tokenized_docs)
# print(f"Loaded {len(documents)} documents into BM25 index.\n")

# # ─────────────────────────────────────────
# # Model
# # ─────────────────────────────────────────
# MODEL_NAME = "google/flan-t5-large"
# print(f"Loading {MODEL_NAME}...")
# tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
# model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
# model.eval()
# print("Model loaded.\n")

# # ─────────────────────────────────────────
# # Prompt Format
# # ─────────────────────────────────────────
# def format_prompt(docs, query, previous_sentences):
#     context = " ".join(docs)
#     history = " ".join(previous_sentences)
#     if history:
#         return (f"Context: {context}\n"
#                 f"Question: {query}\n"
#                 f"Answer so far: {history}\n"
#                 f"Continue the answer in one sentence:")
#     return (f"Context: {context}\n"
#             f"Question: {query}\n"
#             f"Answer in one sentence:")

# # ─────────────────────────────────────────
# # Generate with Token Probabilities
# # ─────────────────────────────────────────
# def generate_with_probs(prompt, max_new_tokens=80):
#     inputs = tokenizer(
#         prompt, return_tensors="pt",
#         truncation=True, max_length=512, padding=True
#     )
#     with torch.no_grad():
#         output = model.generate(
#             input_ids=inputs["input_ids"],
#             attention_mask=inputs["attention_mask"],
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             return_dict_in_generate=True,
#             output_scores=True,
#         )
#     generated_ids = output.sequences[0][1:]
#     token_probs = []
#     for i, score in enumerate(output.scores):
#         probs = torch.softmax(score[0], dim=-1)
#         token_probs.append(probs[generated_ids[i].item()].item())

#     generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
#     return generated_text.strip(), token_probs

# # ─────────────────────────────────────────
# # Retrieve
# # ─────────────────────────────────────────
# def retrieve(query_text, n=3):
#     tokens = query_text.lower().split()
#     if not tokens:
#         return documents[:n]
#     return bm25.get_top_n(tokens, documents, n=n)

# # ─────────────────────────────────────────
# # Extract First Sentence
# # ─────────────────────────────────────────
# def extract_first_sentence(text):
#     sentences = nltk.sent_tokenize(text)
#     return sentences[0].strip() if sentences else text.strip()

# # ─────────────────────────────────────────
# # Mask Low-Confidence Words
# # ─────────────────────────────────────────
# def mask_low_confidence_tokens(sentence, token_probs, beta=0.4):
#     word_tokens = sentence.split()
#     if not word_tokens:
#         return sentence
#     word_prob_map = []
#     subword_idx = 0
#     for word in word_tokens:
#         encoded = tokenizer.encode(word, add_special_tokens=False)
#         n_sub = len(encoded)
#         if subword_idx + n_sub <= len(token_probs):
#             avg_prob = float(np.mean(token_probs[subword_idx: subword_idx + n_sub]))
#         else:
#             avg_prob = 1.0
#         word_prob_map.append((word, avg_prob))
#         subword_idx += n_sub
#     kept = [w for w, p in word_prob_map if p >= beta]
#     return " ".join(kept) if kept else sentence

# # ─────────────────────────────────────────
# # FIX: Smarter completion check
# # Stops on: duplicate sentence, sufficient length,
# # or if the new sentence adds no new information.
# # ─────────────────────────────────────────
# def is_complete(sentences, max_sentences=3):
#     if len(sentences) >= max_sentences:
#         return True
#     if len(sentences) >= 2 and sentences[-1] == sentences[-2]:
#         return True
#     # Stop if a sentence is already a full, standalone answer
#     if sentences:
#         last = sentences[-1]
#         word_count = sum(len(s.split()) for s in sentences)
#         if word_count > 40:
#             return True
#     return False

# # ─────────────────────────────────────────
# # FIX: Deduplicate final answer sentences
# # Prevents "New Delhi. New Delhi is the capital..."
# # ─────────────────────────────────────────
# def deduplicate_sentences(sentences):
#     seen = set()
#     result = []
#     for s in sentences:
#         # Normalise for comparison but keep original for output
#         key = s.lower().strip().rstrip(".")
#         if key not in seen:
#             seen.add(key)
#             result.append(s)
#     return result

# # ─────────────────────────────────────────
# # Exact Match
# # ─────────────────────────────────────────
# def exact_match(prediction, reference):
#     return 1 if reference.lower() in prediction.lower() else 0

# # ─────────────────────────────────────────
# # BASELINE 1: No Retrieval
# # ─────────────────────────────────────────
# def no_retrieval(query):
#     prompt = f"Answer this question in one sentence: {query}"
#     result, _ = generate_with_probs(prompt)
#     return result

# # ─────────────────────────────────────────
# # BASELINE 2: Single-Time RAG
# # ─────────────────────────────────────────
# def rag(query):
#     docs = retrieve(query)
#     prompt = format_prompt(docs, query, [])
#     result, _ = generate_with_probs(prompt)
#     return result

# # ─────────────────────────────────────────
# # FLARE: Forward-Looking Active Retrieval
# # ─────────────────────────────────────────
# def flare(query, theta=0.5, beta=0.4, max_steps=5):
#     """
#     FLAREدirect (paper Section 3.2)
#     theta : min token prob below this → trigger retrieval
#     beta  : words below this prob are masked in query
#     """
#     y = []
#     retrieval_log = []
#     docs = retrieve(query)

#     for step in range(max_steps):
#         prompt = format_prompt(docs, query, y)
#         temp_raw, token_probs = generate_with_probs(prompt)
#         temp_sentence = extract_first_sentence(temp_raw)

#         min_prob = min(token_probs) if token_probs else 1.0
#         avg_prob = float(np.mean(token_probs)) if token_probs else 1.0

#         if min_prob >= theta:
#             y.append(temp_sentence)
#             retrieval_log.append({
#                 "step": step + 1, "triggered": False,
#                 "min_prob": round(min_prob, 3),
#                 "avg_prob": round(avg_prob, 3),
#                 "sentence": temp_sentence,
#             })
#         else:
#             masked_query = mask_low_confidence_tokens(temp_sentence, token_probs, beta)
#             docs = retrieve(masked_query)

#             new_prompt = format_prompt(docs, query, y)
#             new_output, _ = generate_with_probs(new_prompt)
#             final_sentence = extract_first_sentence(new_output)

#             y.append(final_sentence)
#             retrieval_log.append({
#                 "step": step + 1, "triggered": True,
#                 "min_prob": round(min_prob, 3),
#                 "avg_prob": round(avg_prob, 3),
#                 "masked_query": masked_query,
#                 "temp_sentence": temp_sentence,
#                 "final_sentence": final_sentence,
#             })

#         if is_complete(y):
#             break

#     # Deduplicate before returning
#     y = deduplicate_sentences(y)
#     return " ".join(y), retrieval_log

# # ─────────────────────────────────────────
# # Print Retrieval Log
# # ─────────────────────────────────────────
# def print_retrieval_log(log):
#     print("\n  [FLARE Retrieval Log]")
#     for entry in log:
#         s = entry["step"]
#         if entry["triggered"]:
#             print(f"  Step {s}: RETRIEVAL TRIGGERED "
#                   f"(min_prob={entry['min_prob']}, avg_prob={entry['avg_prob']})")
#             print(f"    temp      : {entry['temp_sentence']}")
#             print(f"    query used: {entry['masked_query']}")
#             print(f"    final     : {entry['final_sentence']}")
#         else:
#             print(f"  Step {s}: accepted "
#                   f"(min_prob={entry['min_prob']}, avg_prob={entry['avg_prob']})")
#             print(f"    sentence  : {entry['sentence']}")

# # ─────────────────────────────────────────
# # Test Cases — expanded for better demo
# # ─────────────────────────────────────────
# test_cases = [
#     {"query": "Who is Joe Biden and where did he study?",
#      "reference": "University of Delaware"},
#     {"query": "What is machine learning?",
#      "reference": "machine learning"},
#     {"query": "What is the capital of India?",
#      "reference": "New Delhi"},
#     {"query": "When did India gain independence?",
#      "reference": "1947"},
#     {"query": "Who created Python programming language?",
#      "reference": "Guido van Rossum"},
#     {"query": "What is deep learning?",
#      "reference": "neural network"},
# ]

# print("=" * 60)
# print(f"FLARE vs RAG vs No Retrieval  [{MODEL_NAME}]")
# print("=" * 60)

# results = {"no_retrieval": [], "rag": [], "flare": []}

# for tc in test_cases:
#     q, ref = tc["query"], tc["reference"]
#     print(f"\nQuery     : {q}")
#     print(f"Reference : '{ref}'")
#     print("-" * 55)

#     nr      = no_retrieval(q)
#     r       = rag(q)
#     fl, log = flare(q)

#     em_nr = exact_match(nr, ref)
#     em_r  = exact_match(r,  ref)
#     em_fl = exact_match(fl, ref)

#     results["no_retrieval"].append(em_nr)
#     results["rag"].append(em_r)
#     results["flare"].append(em_fl)

#     print(f"No Retrieval [{em_nr}] : {nr[:110]}")
#     print(f"RAG          [{em_r}] : {r[:110]}")
#     print(f"FLARE        [{em_fl}] : {fl[:110]}")
#     print_retrieval_log(log)

# print("\n" + "=" * 60)
# print("Overall Exact Match Scores")
# print("=" * 60)
# for method, scores in results.items():
#     avg = round(sum(scores) / len(scores), 2)
#     bar = "█" * int(avg * 20)
#     print(f"  {method:<20} EM = {avg}  {bar}  raw: {scores}")

# import nltk
# import torch
# import numpy as np
# from rank_bm25 import BM25Okapi
# from transformers import T5ForConditionalGeneration, T5Tokenizer

# nltk.download('punkt', quiet=True)
# nltk.download('punkt_tab', quiet=True)

# # ─────────────────────────────────────────
# # Load Dataset
# # ─────────────────────────────────────────
# def load_documents(file_path):
#     with open(file_path, "r") as f:
#         docs = f.read().split("\n")
#     return [doc for doc in docs if doc.strip() != ""]

# documents = load_documents("data.txt")
# tokenized_docs = [doc.lower().split() for doc in documents]
# bm25 = BM25Okapi(tokenized_docs)
# print(f"Loaded {len(documents)} documents into BM25 index.\n")

# MODEL_NAME = "google/flan-t5-large"
# print(f"Loading {MODEL_NAME}...")
# tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
# model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
# model.eval()
# print("Model loaded.\n")

# # ─────────────────────────────────────────
# # Prompt Format
# # ─────────────────────────────────────────
# def format_prompt(docs, query, previous_sentences):
#     context = " ".join(docs)
#     history = " ".join(previous_sentences)
#     if history:
#         return (f"Context: {context}\n"
#                 f"Question: {query}\n"
#                 f"Answer so far: {history}\n"
#                 f"Continue the answer in one sentence:")
#     return (f"Context: {context}\n"
#             f"Question: {query}\n"
#             f"Answer in one sentence:")

# # ─────────────────────────────────────────
# # Generate with Token Probabilities
# # ─────────────────────────────────────────
# def generate_with_probs(prompt, max_new_tokens=80):
#     inputs = tokenizer(
#         prompt, return_tensors="pt",
#         truncation=True, max_length=512, padding=True
#     )
#     with torch.no_grad():
#         output = model.generate(
#             input_ids=inputs["input_ids"],
#             attention_mask=inputs["attention_mask"],
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             return_dict_in_generate=True,
#             output_scores=True,
#         )
#     generated_ids = output.sequences[0][1:]
#     token_probs = []
#     for i, score in enumerate(output.scores):
#         probs = torch.softmax(score[0], dim=-1)
#         token_probs.append(probs[generated_ids[i].item()].item())
#     generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
#     return generated_text.strip(), token_probs

# # ─────────────────────────────────────────
# # FIX 1: Fallback when model says "unanswerable"
# # Flan-T5 outputs "unanswerable" when it cannot
# # find the answer in context. In that case, we
# # retry WITHOUT context so the model uses its
# # own parametric knowledge instead.
# # ─────────────────────────────────────────
# def generate_with_fallback(prompt, query, token_probs_holder):
#     text, probs = generate_with_probs(prompt)
#     token_probs_holder.extend(probs)

#     if text.lower().strip() in ("unanswerable", "unknown", "i don't know", ""):
#         fallback_prompt = f"Answer this question in one sentence: {query}"
#         text, probs = generate_with_probs(fallback_prompt)
#         token_probs_holder.clear()
#         token_probs_holder.extend(probs)

#     return text

# # ─────────────────────────────────────────
# # Retrieve
# # ─────────────────────────────────────────
# def retrieve(query_text, n=3):
#     tokens = query_text.lower().split()
#     if not tokens:
#         return documents[:n]
#     return bm25.get_top_n(tokens, documents, n=n)

# # ─────────────────────────────────────────
# # Extract First Sentence
# # ─────────────────────────────────────────
# def extract_first_sentence(text):
#     sentences = nltk.sent_tokenize(text)
#     return sentences[0].strip() if sentences else text.strip()

# # ─────────────────────────────────────────
# # Mask Low-Confidence Words
# # ─────────────────────────────────────────
# def mask_low_confidence_tokens(sentence, token_probs, beta=0.4):
#     word_tokens = sentence.split()
#     if not word_tokens:
#         return sentence
#     word_prob_map = []
#     subword_idx = 0
#     for word in word_tokens:
#         encoded = tokenizer.encode(word, add_special_tokens=False)
#         n_sub = len(encoded)
#         if subword_idx + n_sub <= len(token_probs):
#             avg_prob = float(np.mean(token_probs[subword_idx: subword_idx + n_sub]))
#         else:
#             avg_prob = 1.0
#         word_prob_map.append((word, avg_prob))
#         subword_idx += n_sub
#     kept = [w for w, p in word_prob_map if p >= beta]
#     return " ".join(kept) if kept else sentence

# # ─────────────────────────────────────────
# # FIX 2: Substring-aware deduplication
# # "New Delhi" + "New Delhi is the capital..."
# # — the second contains the first, so drop
# # the shorter one and keep the richer sentence.
# # ─────────────────────────────────────────
# def deduplicate_sentences(sentences):
#     cleaned = [s.lower().strip().rstrip(".") for s in sentences]
#     result = []
#     for i, s in enumerate(sentences):
#         c = cleaned[i]
#         # Keep this sentence only if no other sentence already contains it
#         dominated = any(
#             c != cleaned[j] and c in cleaned[j]
#             for j in range(len(sentences)) if j != i
#         )
#         if not dominated and c not in [cleaned[k] for k in range(i)]:
#             result.append(s)
#     return result if result else [sentences[-1]]

# # ─────────────────────────────────────────
# # FIX 3: Smarter completion — stop if the
# # current answer already contains the key
# # facts (high avg_prob + short answer is fine)
# # ─────────────────────────────────────────
# def is_complete(sentences, max_sentences=3):
#     if len(sentences) >= max_sentences:
#         return True
#     if len(sentences) >= 2 and sentences[-1] == sentences[-2]:
#         return True
#     if sentences:
#         word_count = sum(len(s.split()) for s in sentences)
#         if word_count > 40:
#             return True
#         # A short confident answer (date, name, place) is already complete
#         last = sentences[-1].strip().rstrip(".")
#         words = last.split()
#         if len(words) <= 6:
#             return True
#     return False

# # ─────────────────────────────────────────
# # Exact Match
# # ─────────────────────────────────────────
# def exact_match(prediction, reference):
#     return 1 if reference.lower() in prediction.lower() else 0

# # ─────────────────────────────────────────
# # BASELINE 1: No Retrieval
# # ─────────────────────────────────────────
# def no_retrieval(query):
#     prompt = f"Answer this question in one sentence: {query}"
#     result, _ = generate_with_probs(prompt)
#     return result

# # ─────────────────────────────────────────
# # BASELINE 2: Single-Time RAG
# # ─────────────────────────────────────────
# def rag(query):
#     docs = retrieve(query)
#     prompt = format_prompt(docs, query, [])
#     probs_holder = []
#     result = generate_with_fallback(prompt, query, probs_holder)
#     return result

# # ─────────────────────────────────────────
# # FLARE: Forward-Looking Active Retrieval
# # ─────────────────────────────────────────
# def flare(query, theta=0.5, beta=0.4, max_steps=5):
#     y = []
#     retrieval_log = []
#     docs = retrieve(query)

#     for step in range(max_steps):
#         prompt = format_prompt(docs, query, y)
#         token_probs = []
#         temp_raw = generate_with_fallback(prompt, query, token_probs)
#         temp_sentence = extract_first_sentence(temp_raw)

#         min_prob = min(token_probs) if token_probs else 1.0
#         avg_prob = float(np.mean(token_probs)) if token_probs else 1.0

#         if min_prob >= theta:
#             y.append(temp_sentence)
#             retrieval_log.append({
#                 "step": step + 1, "triggered": False,
#                 "min_prob": round(min_prob, 3),
#                 "avg_prob": round(avg_prob, 3),
#                 "sentence": temp_sentence,
#             })
#         else:
#             masked_query = mask_low_confidence_tokens(temp_sentence, token_probs, beta)
#             docs = retrieve(masked_query)

#             new_prompt = format_prompt(docs, query, y)
#             new_probs = []
#             new_output = generate_with_fallback(new_prompt, query, new_probs)
#             final_sentence = extract_first_sentence(new_output)

#             y.append(final_sentence)
#             retrieval_log.append({
#                 "step": step + 1, "triggered": True,
#                 "min_prob": round(min_prob, 3),
#                 "avg_prob": round(avg_prob, 3),
#                 "masked_query": masked_query,
#                 "temp_sentence": temp_sentence,
#                 "final_sentence": final_sentence,
#             })

#         if is_complete(y):
#             break

#     y = deduplicate_sentences(y)
#     return " ".join(y), retrieval_log

# # ─────────────────────────────────────────
# # Print Retrieval Log
# # ─────────────────────────────────────────
# def print_retrieval_log(log):
#     print("\n  [FLARE Retrieval Log]")
#     for entry in log:
#         s = entry["step"]
#         if entry["triggered"]:
#             print(f"  Step {s}: RETRIEVAL TRIGGERED "
#                   f"(min_prob={entry['min_prob']}, avg_prob={entry['avg_prob']})")
#             print(f"    temp      : {entry['temp_sentence']}")
#             print(f"    query used: {entry['masked_query']}")
#             print(f"    final     : {entry['final_sentence']}")
#         else:
#             print(f"  Step {s}: accepted "
#                   f"(min_prob={entry['min_prob']}, avg_prob={entry['avg_prob']})")
#             print(f"    sentence  : {entry['sentence']}")

# # ─────────────────────────────────────────
# # Test Cases
# # ─────────────────────────────────────────
# test_cases = [
#     {"query": "Who is Joe Biden and where did he study?",
#      "reference": "University of Delaware"},
#     {"query": "What is machine learning?",
#      "reference": "machine learning"},
#     {"query": "What is the capital of India?",
#      "reference": "New Delhi"},
#     {"query": "When did India gain independence?",
#      "reference": "1947"},
#     {"query": "Who created Python programming language?",
#      "reference": "Guido van Rossum"},
#     {"query": "What is deep learning?",
#      "reference": "neural network"},
# ]

# print("=" * 60)
# print(f"FLARE vs RAG vs No Retrieval  [{MODEL_NAME}]")
# print("=" * 60)

# results = {"no_retrieval": [], "rag": [], "flare": []}

# for tc in test_cases:
#     q, ref = tc["query"], tc["reference"]
#     print(f"\nQuery     : {q}")
#     print(f"Reference : '{ref}'")
#     print("-" * 55)

#     nr      = no_retrieval(q)
#     r       = rag(q)
#     fl, log = flare(q)

#     em_nr = exact_match(nr, ref)
#     em_r  = exact_match(r,  ref)
#     em_fl = exact_match(fl, ref)

#     results["no_retrieval"].append(em_nr)
#     results["rag"].append(em_r)
#     results["flare"].append(em_fl)

#     print(f"No Retrieval [{em_nr}] : {nr[:110]}")
#     print(f"RAG          [{em_r}] : {r[:110]}")
#     print(f"FLARE        [{em_fl}] : {fl[:110]}")
#     print_retrieval_log(log)

# print("\n" + "=" * 60)
# print("Overall Exact Match Scores")
# print("=" * 60)
# for method, scores in results.items():
#     avg = round(sum(scores) / len(scores), 2)
#     bar = "█" * int(avg * 20)
#     print(f"  {method:<20} EM = {avg}  {bar}  raw: {scores}")

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