# evaluate.py — Objective 4 Evaluation Suite
#
# Implements the full evaluation framework described in Section 3.6 of the report:
#   - Retrieval quality     : Precision@k and Recall@k (k=3 and k=5)
#   - Response faithfulness : Gemini-scored (RAGAs-style)
#   - Answer relevancy      : Gemini-scored (RAGAs-style)
#   - Context precision     : Gemini-scored (RAGAs-style)
#   - Response latency      : median, mean, 95th percentile
#   - Robustness            : metric variance across 3 surface-form variants (L1 only)
#   - Baseline comparison   : Gemini 2.5 Flash with NO retrieval augmentation
#
# Usage:
#   python evaluate.py
#   python evaluate.py --dataset test_dataset.json --output eval_results.json
#
# Prerequisites:
#   pip install google-genai chromadb python-dotenv numpy
#   .env must contain GOOGLE_API_KEY=<your key>
#   Documents must already be indexed (run ingest.py first)

import os
import re
import sys
import json
import time
import logging
import argparse
import statistics
from datetime import datetime
from dotenv import load_dotenv
from google import genai
import chromadb

# ── Import from existing pipeline ─────────────────────────────────────────────
# We reuse the retrieval and embedding functions already written in query.py
# so evaluation runs over exactly the same pipeline the chatbot uses in production.
from query import retrieve, get_embedding, compute_confidence, answer, build_context

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename='evaluation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ── Gemini client for LLM-scored metrics ──────────────────────────────────────
# A separate client is used here so the evaluator does not interfere with the
# production clients in query.py.
client_eval = genai.Client(
    api_key=os.getenv('GOOGLE_API_KEY'),
    http_options={'api_version': 'v1'}
)

EVAL_MODEL   = 'models/gemini-2.5-flash'   # Model used for all scoring prompts
DATASET_PATH = 'test_dataset.json'          # Default dataset location
OUTPUT_PATH  = 'eval_results.json'          # Where results are written


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — RETRIEVAL METRICS (Precision@k, Recall@k)
# ══════════════════════════════════════════════════════════════════════════════

def precision_at_k(retrieved_sources, ground_truth_sources, k):
    """
    Precision@k = (number of retrieved chunks whose source is in ground truth) / k

    retrieved_sources    : list of source filenames from top-k chunks (in rank order)
    ground_truth_sources : list of correct source filenames from the test dataset
    k                    : cutoff rank

    Returns a float between 0.0 and 1.0.
    """
    top_k  = retrieved_sources[:k]
    gt_set = set(ground_truth_sources)
    hits   = sum(1 for src in top_k if src in gt_set)
    return round(hits / k, 4) if k > 0 else 0.0


def recall_at_k(retrieved_sources, ground_truth_sources, k):
    """
    Recall@k = (number of ground-truth sources found in top-k) / total ground-truth sources

    A ground-truth source is considered 'found' if at least one retrieved chunk
    in the top-k belongs to that source file.

    Returns a float between 0.0 and 1.0.
    """
    top_k  = set(retrieved_sources[:k])
    gt_set = set(ground_truth_sources)
    found  = gt_set.intersection(top_k)
    return round(len(found) / len(gt_set), 4) if gt_set else 0.0


def run_retrieval_metrics(query, ground_truth_sources, k_values=(3, 5), retries=3):
    """
    Run the retrieval pipeline for a query and compute Precision and Recall
    at each specified k value. Retries on network errors.
    """
    for attempt in range(retries):
        try:
            chunks, avg_conf = retrieve(query, k=max(k_values))
            break
        except Exception as e:
            err = str(e)
            if ('getaddrinfo' in err or 'ConnectError' in err or 'Connection' in err) and attempt < retries - 1:
                wait = (attempt + 1) * 30
                print(f'     Network error on attempt {attempt+1}. Waiting {wait}s...')
                time.sleep(wait)
            else:
                logging.error(f'Retrieval error: {e}')
                return {
                    'retrieved_sources': [], 'avg_confidence': 0.0,
                    'precision@3': 0.0, 'recall@3': 0.0,
                    'precision@5': 0.0, 'recall@5': 0.0,
                }
    else:
        return {
            'retrieved_sources': [], 'avg_confidence': 0.0,
            'precision@3': 0.0, 'recall@3': 0.0,
            'precision@5': 0.0, 'recall@5': 0.0,
        }

    # Extract source filenames in rank order (chunks already sorted by score desc)
    retrieved_sources = [c['meta'].get('source', 'Unknown') for c in chunks]

    result = {
        'retrieved_sources': retrieved_sources,
        'avg_confidence':    round(avg_conf, 4),
    }
    for k in k_values:
        result[f'precision@{k}'] = precision_at_k(retrieved_sources, ground_truth_sources, k)
        result[f'recall@{k}']    = recall_at_k(retrieved_sources, ground_truth_sources, k)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LLM-SCORED METRICS (RAGAs-style)
# These three functions send structured prompts to Gemini and parse a score.
# This mirrors the approach used by the RAGAs framework (Es et al., 2024),
# adapted for Gemini and implemented without the ragas library dependency.
# ══════════════════════════════════════════════════════════════════════════════

def _call_gemini_score(prompt, retries=3):
    """
    Helper that sends a scoring prompt to Gemini and extracts a float 0-1.
    Returns None if parsing fails after all retries.
    """
    for attempt in range(retries):
        try:
            response = client_eval.models.generate_content(
                model=EVAL_MODEL,
                contents=prompt
            )
            text = response.text.strip()
            # Extract the first decimal or integer number from the response
            match = re.search(r'\b(0(\.\d+)?|1(\.0+)?)\b', text)
            if match:
                return round(float(match.group()), 4)
            logging.warning(f'Score parse failed. Response: {text[:200]}')
            return None
        except Exception as e:
            if '429' in str(e) and attempt < retries - 1:
                logging.info('Rate limited on scoring call. Waiting 65s...')
                time.sleep(65)
            else:
                logging.error(f'Gemini scoring error: {e}')
                return None
    return None


def score_faithfulness(question, context, answer_text):
    """
    Faithfulness: does the answer contain only claims supported by the context?

    Score 1.0 = every claim supported by retrieved context.
    Score 0.0 = answer contains claims not present in context.

    Prompt design follows the RAGAs faithfulness metric (Es et al., 2024).
    """
    prompt = f"""You are an impartial evaluator assessing whether an AI answer is faithful to its source context.

CONTEXT (retrieved policy passages):
{context}

QUESTION: {question}

ANSWER: {answer_text}

TASK:
1. Identify every factual claim made in the ANSWER.
2. For each claim, check whether it is directly supported by the CONTEXT.
3. Compute: faithfulness = (claims supported by context) / (total claims in answer).

Return ONLY a single number between 0 and 1 (e.g. 0.85).
Do not include explanation. Do not include any other text."""
    return _call_gemini_score(prompt)


def score_answer_relevancy(question, answer_text):
    """
    Answer Relevancy: how well does the answer address the question asked?

    Score 1.0 = fully relevant and complete.
    Score 0.0 = completely unrelated or a refusal.

    Simplified version of RAGAs answer relevancy metric.
    """
    prompt = f"""You are an impartial evaluator assessing how well an AI answer addresses a student question.

QUESTION: {question}

ANSWER: {answer_text}

TASK:
Score how relevant the ANSWER is to the QUESTION.
- 1.0 = directly and completely addresses what was asked
- 0.5 = partially relevant but misses key aspects
- 0.0 = does not address the question at all

Return ONLY a single number between 0 and 1 (e.g. 0.9).
Do not include explanation. Do not include any other text."""
    return _call_gemini_score(prompt)


def score_context_precision(question, chunks):
    """
    Context Precision: what fraction of retrieved chunks are relevant to the question?

    Score 1.0 = every retrieved chunk was useful.
    Score 0.0 = no retrieved chunk was relevant.

    Reflects signal-to-noise ratio of the retrieval step.
    """
    chunk_texts = '\n\n---\n\n'.join([
        f'[Chunk {i+1} | Source: {c["meta"].get("source","?")}]\n{c["text"][:300]}'
        for i, c in enumerate(chunks[:5])
    ])
    prompt = f"""You are an impartial evaluator assessing retrieved document chunks.

QUESTION: {question}

RETRIEVED CHUNKS:
{chunk_texts}

TASK:
For each chunk, decide whether it contains information relevant to answering the QUESTION.
Compute: context precision = (number of relevant chunks) / (total chunks shown).

Return ONLY a single number between 0 and 1 (e.g. 0.6).
Do not include explanation. Do not include any other text."""
    return _call_gemini_score(prompt)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BASELINE (Gemini WITHOUT retrieval augmentation)
# Queries Gemini directly with no context for comparison against the RAG system.
# ══════════════════════════════════════════════════════════════════════════════

BASELINE_SYSTEM_PROMPT = """You are an academic policy assistant for Middlesex University Mauritius.
Answer the student question using your general knowledge only.
You do NOT have access to any specific policy documents.
Be honest if you are unsure."""

def run_baseline(query, retries=3):
    """
    Run the same query through Gemini 2.5 Flash with NO retrieved context.
    Returns (response_text, elapsed_seconds). Retries on network/rate errors.
    """
    prompt = f"""{BASELINE_SYSTEM_PROMPT}

Student question: {query}

Answer:"""
    start = time.time()
    for attempt in range(retries):
        try:
            response = client_eval.models.generate_content(
                model=EVAL_MODEL,
                contents=prompt
            )
            return response.text, round(time.time() - start, 2)
        except Exception as e:
            err = str(e)
            if '429' in err and attempt < retries - 1:
                time.sleep(65)
            elif ('getaddrinfo' in err or 'ConnectError' in err or 'Connection' in err) and attempt < retries - 1:
                wait = (attempt + 1) * 30
                print(f'     Network error on baseline attempt {attempt+1}. Waiting {wait}s...')
                time.sleep(wait)
            else:
                logging.error(f'Baseline error: {e}')
                return '', round(time.time() - start, 2)
    return '', round(time.time() - start, 2)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ROBUSTNESS TESTING (L1 queries only)
# Each Level 1 query is submitted in 4 forms: original + paraphrase +
# keyword-reduced + typographically noisy. Precision@5 variance across the
# four forms measures how robust the retrieval step is to surface variation.
# ══════════════════════════════════════════════════════════════════════════════

def run_robustness(item, delay=2.0):
    """
    Run all surface variants for a Level 1 query and compute variance in
    Precision@5 across the variants. Returns a dict with per-variant scores
    and summary statistics.
    """
    v = item.get('variants', {})
    variants = {
        'original':        item.get('query') or item.get('question', ''),
        'paraphrase':      v.get('paraphrase', ''),
        'keyword_reduced': v.get('keyword_reduced', ''),
        'noisy':           v.get('noisy') or v.get('typo') or v.get('noise', ''),
    }
    # Skip any variant that is empty
    variants = {k: val for k, val in variants.items() if val}

    precision_scores = {}
    for label, query_text in variants.items():
        ret = run_retrieval_metrics(query_text, item['ground_truth_sources'])
        precision_scores[label] = ret['precision@5']
        time.sleep(delay)

    scores      = list(precision_scores.values())
    score_var   = round(statistics.variance(scores), 6) if len(scores) > 1 else 0.0
    score_range = round(max(scores) - min(scores), 4)

    return {
        'precision@5_per_variant': precision_scores,
        'variance':                score_var,
        'range':                   score_range,
        'robust':                  score_range < 0.10,  # Report target: < 10% range
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(dataset_path=DATASET_PATH, output_path=OUTPUT_PATH, delay=3.0):
    """
    Main evaluation entry point. Runs all metrics for every query in the dataset
    and writes a structured JSON report plus a printed summary table.
    """
    print(f'\n{"="*60}')
    print('  MUM RAG Chatbot — Objective 4 Evaluation')
    print(f'  Dataset : {dataset_path}')
    print(f'  Output  : {output_path}')
    print(f'  Started : {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print(f'{"="*60}\n')

    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw = f.read()

    # Handle plain array, dict with 'queries' key, or other wrappers
    dataset = json.loads(raw)
    if isinstance(dataset, str):
        dataset = json.loads(dataset)
    if isinstance(dataset, dict):
        # Prefer 'queries' key, then fall back to first list value
        if 'queries' in dataset:
            dataset = dataset['queries']
        else:
            for v in dataset.values():
                if isinstance(v, list):
                    dataset = v
                    break
            else:
                print(f'ERROR: Could not find a list inside the dict in {dataset_path}')
                sys.exit(1)
    if not isinstance(dataset, list):
        print(f'ERROR: Expected a JSON array in {dataset_path}, got {type(dataset).__name__}')
        sys.exit(1)

    # ── Resume support ────────────────────────────────────────────────────────
    # If output_path already exists, load completed results and skip those queries.
    results       = []
    rag_latencies = []
    completed_ids = set()

    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            if 'results' in existing and existing['results']:
                results       = existing['results']
                completed_ids = {r['id'] for r in results}
                rag_latencies = [r['rag']['latency_s'] for r in results if r.get('rag', {}).get('latency_s')]
                print(f'  Resuming — {len(completed_ids)} queries already completed: {sorted(completed_ids)}')
                print()
        except Exception:
            pass

    for i, item in enumerate(dataset, 1):
        qid   = item['id']
        level = item['level']
        query = item.get('query') or item.get('question', '')
        gt    = item.get('ground_truth_sources', [])

        if qid in completed_ids:
            print(f'[{i:02d}/{len(dataset)}] {qid}  — already completed, skipping')
            continue

        print(f'[{i:02d}/{len(dataset)}] {qid}  (Level {level})')
        print(f'  Q: {query[:90]}' + ('...' if len(query) > 90 else ''))

        record = {'id': qid, 'level': level, 'query': query}

        # ── 1. Retrieval metrics ──────────────────────────────────────────────
        print('  → Retrieval metrics...')
        ret_metrics = run_retrieval_metrics(query, gt)
        record['retrieval'] = ret_metrics
        print(f'     P@3={ret_metrics["precision@3"]:.2f}  '
              f'P@5={ret_metrics["precision@5"]:.2f}  '
              f'R@3={ret_metrics["recall@3"]:.2f}  '
              f'R@5={ret_metrics["recall@5"]:.2f}  '
              f'Conf={ret_metrics["avg_confidence"]:.2f}')
        time.sleep(delay)

        # ── 2. Full RAG answer + latency ──────────────────────────────────────
        print('  → Running RAG answer...')
        t0 = time.time()
        try:
            rag_response, _, rag_score, rag_qtype = answer(query, length='Standard')
            rag_elapsed = round(time.time() - t0, 2)
        except Exception as e:
            rag_response = ''
            rag_elapsed  = round(time.time() - t0, 2)
            rag_qtype    = 'error'
            logging.error(f'RAG answer error for {qid}: {e}')

        rag_latencies.append(rag_elapsed)
        record['rag'] = {
            'response':   rag_response[:500],
            'latency_s':  rag_elapsed,
            'query_type': rag_qtype,
        }
        print(f'     Latency: {rag_elapsed}s  Type: {rag_qtype}')
        time.sleep(delay)

        # ── 3. LLM-scored metrics (RAGAs-style) ───────────────────────────────
        if rag_response:
            chunks, _ = retrieve(query, k=5)
            context   = build_context(chunks)

            print('  → Scoring faithfulness...')
            faithfulness = score_faithfulness(query, context, rag_response)
            time.sleep(delay)

            print('  → Scoring answer relevancy...')
            relevancy = score_answer_relevancy(query, rag_response)
            time.sleep(delay)

            print('  → Scoring context precision...')
            ctx_precision = score_context_precision(query, chunks)
            time.sleep(delay)
        else:
            chunks        = []
            context       = ''
            faithfulness  = None
            relevancy     = None
            ctx_precision = None

        record['llm_scores'] = {
            'faithfulness':      faithfulness,
            'answer_relevancy':  relevancy,
            'context_precision': ctx_precision,
        }
        print(f'     Faith={faithfulness}  Rel={relevancy}  CtxPrec={ctx_precision}')

        # ── 4. Baseline (no retrieval) ────────────────────────────────────────
        print('  → Running baseline...')
        baseline_text, baseline_elapsed = run_baseline(query)
        time.sleep(delay)

        if baseline_text and context:
            print('  → Scoring baseline faithfulness...')
            baseline_faith = score_faithfulness(query, context, baseline_text)
            time.sleep(delay)
        else:
            baseline_faith = None

        record['baseline'] = {
            'response':     baseline_text[:500],
            'latency_s':    baseline_elapsed,
            'faithfulness': baseline_faith,
        }
        print(f'     Baseline faith={baseline_faith}  Latency={baseline_elapsed}s')

        # ── 5. Robustness (L1 only) ───────────────────────────────────────────
        if level == 1 and 'variants' in item:
            print('  → Robustness testing...')
            robustness = run_robustness(item, delay=delay)
            record['robustness'] = robustness
            print(f'     Variance={robustness["variance"]:.4f}  '
                  f'Range={robustness["range"]:.2f}  '
                  f'Robust={robustness["robust"]}')

        results.append(record)

        # Save partial results after every query so a crash can be resumed
        partial_output = {
            'run_metadata': {
                'timestamp':     datetime.now().isoformat(),
                'dataset':       dataset_path,
                'model':         EVAL_MODEL,
                'total_queries': len(dataset),
                'completed':     len(results),
            },
            'results': results,
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(partial_output, f, indent=2)
        print()

    # ── Aggregate and save ────────────────────────────────────────────────────
    summary = _compute_summary(results, rag_latencies)
    output  = {
        'run_metadata': {
            'timestamp':     datetime.now().isoformat(),
            'dataset':       dataset_path,
            'model':         EVAL_MODEL,
            'total_queries': len(dataset),
        },
        'summary': summary,
        'results': results,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    _print_summary(summary)
    print(f'\nFull results saved to: {output_path}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — AGGREGATION AND REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def _safe_avg(values):
    filtered = [v for v in values if v is not None]
    return round(sum(filtered) / len(filtered), 4) if filtered else None


def _compute_summary(results, rag_latencies):
    by_level = {1: [], 2: [], 3: []}
    for r in results:
        by_level[r['level']].append(r)

    def level_stats(items):
        if not items:
            return {}
        return {
            'n':                        len(items),
            'avg_precision@3':          _safe_avg([i['retrieval']['precision@3'] for i in items]),
            'avg_precision@5':          _safe_avg([i['retrieval']['precision@5'] for i in items]),
            'avg_recall@3':             _safe_avg([i['retrieval']['recall@3'] for i in items]),
            'avg_recall@5':             _safe_avg([i['retrieval']['recall@5'] for i in items]),
            'avg_faithfulness':         _safe_avg([i['llm_scores']['faithfulness'] for i in items]),
            'avg_answer_relevancy':     _safe_avg([i['llm_scores']['answer_relevancy'] for i in items]),
            'avg_context_precision':    _safe_avg([i['llm_scores']['context_precision'] for i in items]),
            'avg_rag_latency_s':        _safe_avg([i['rag']['latency_s'] for i in items]),
            'avg_baseline_faithfulness':_safe_avg([
                i['baseline']['faithfulness'] for i in items
                if i['baseline'].get('faithfulness') is not None
            ]),
        }

    l1_robust = [r for r in results if r['level'] == 1 and 'robustness' in r]
    robustness_summary = {
        'n_tested':    len(l1_robust),
        'n_robust':    sum(1 for r in l1_robust if r['robustness']['robust']),
        'avg_variance': _safe_avg([r['robustness']['variance'] for r in l1_robust]),
        'avg_range':    _safe_avg([r['robustness']['range'] for r in l1_robust]),
        'pct_robust':   round(
            sum(1 for r in l1_robust if r['robustness']['robust']) / len(l1_robust) * 100, 1
        ) if l1_robust else None,
    }

    if rag_latencies:
        sorted_lat = sorted(rag_latencies)
        n95 = max(1, int(len(sorted_lat) * 0.95))
        latency_dist = {
            'median_s': round(statistics.median(sorted_lat), 2),
            'mean_s':   round(statistics.mean(sorted_lat), 2),
            'p95_s':    round(sorted_lat[n95 - 1], 2),
            'min_s':    round(min(sorted_lat), 2),
            'max_s':    round(max(sorted_lat), 2),
        }
    else:
        latency_dist = {}

    return {
        'overall':    level_stats(results),
        'by_level':   {f'L{k}': level_stats(v) for k, v in by_level.items()},
        'robustness': robustness_summary,
        'latency':    latency_dist,
    }


def _print_summary(summary):
    W = 60
    print('\n' + '=' * W)
    print('  EVALUATION SUMMARY')
    print('=' * W)

    targets = {
        'precision@5': 0.70, 'recall@5': 0.75,
        'faithfulness': 0.85, 'answer_relevancy': 0.80,
        'context_precision': 0.70,
    }

    def row(label, value, target=None):
        if value is None:
            print(f'  {label:<38} N/A')
            return
        flag = ''
        if target is not None:
            flag = '  ✓' if value >= target else f'  ✗ (target {target:.2f})'
        print(f'  {label:<38} {value:.4f}{flag}')

    o = summary['overall']
    print('\n  OVERALL (all queries)')
    print('  ' + '-' * (W - 2))
    row('Precision@3',            o.get('avg_precision@3'),          targets['precision@5'])
    row('Precision@5',            o.get('avg_precision@5'),          targets['precision@5'])
    row('Recall@3',               o.get('avg_recall@3'),             targets['recall@5'])
    row('Recall@5',               o.get('avg_recall@5'),             targets['recall@5'])
    row('Faithfulness  (RAG)',     o.get('avg_faithfulness'),         targets['faithfulness'])
    row('Answer Relevancy',        o.get('avg_answer_relevancy'),    targets['answer_relevancy'])
    row('Context Precision',       o.get('avg_context_precision'),   targets['context_precision'])
    row('Faithfulness  (Baseline)',o.get('avg_baseline_faithfulness'))

    print('\n  PER DIFFICULTY LEVEL')
    print('  ' + '-' * (W - 2))
    for lk in ['L1', 'L2', 'L3']:
        lv = summary['by_level'].get(lk, {})
        print(f'\n  {lk} ({lv.get("n", 0)} queries)')
        row('  Precision@5',     lv.get('avg_precision@5'))
        row('  Recall@5',        lv.get('avg_recall@5'))
        row('  Faithfulness',    lv.get('avg_faithfulness'))
        row('  Answer Relevancy',lv.get('avg_answer_relevancy'))
        row('  Ctx Precision',   lv.get('avg_context_precision'))

    print('\n  LATENCY DISTRIBUTION')
    print('  ' + '-' * (W - 2))
    lat = summary.get('latency', {})
    row('Median (s)',   lat.get('median_s'))
    row('Mean (s)',     lat.get('mean_s'))
    row('P95 (s)',      lat.get('p95_s'),  target=10.0)
    row('Min (s)',      lat.get('min_s'))
    row('Max (s)',      lat.get('max_s'))

    print('\n  ROBUSTNESS  (L1 queries only)')
    print('  ' + '-' * (W - 2))
    rob = summary.get('robustness', {})
    print(f'  {"Queries tested":<38} {rob.get("n_tested", "N/A")}')
    print(f'  {"Robust (range < 10%)":<38} {rob.get("n_robust", "N/A")}'
          f' / {rob.get("n_tested", "N/A")}  ({rob.get("pct_robust", "N/A")}%)')
    row('Avg P@5 variance', rob.get('avg_variance'))
    row('Avg P@5 range',    rob.get('avg_range'))
    print('\n' + '=' * W)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Objective 4 Evaluation Suite — MUM RAG Chatbot'
    )
    parser.add_argument('--dataset', default=DATASET_PATH,
                        help=f'Path to test dataset JSON (default: {DATASET_PATH})')
    parser.add_argument('--output',  default=OUTPUT_PATH,
                        help=f'Path to save results JSON (default: {OUTPUT_PATH})')
    parser.add_argument('--delay',   type=float, default=3.0,
                        help='Seconds between API calls (default: 3.0)')
    args = parser.parse_args()

    evaluate(
        dataset_path=args.dataset,
        output_path=args.output,
        delay=args.delay,
    )