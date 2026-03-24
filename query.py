# query.py - Query Processing Pipeline
import os
import re
import time
import logging
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    filename='chatbot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

COLLECTION     = 'mum_policy_docs'
TOP_K          = 8
MIN_CONFIDENCE = 0.08
DB_PATH        = './chroma_store'

# Client for embeddings — requires v1beta
client_embed = genai.Client(
    api_key=os.getenv('GOOGLE_API_KEY'),
    http_options={'api_version': 'v1beta'}
)

# Client for text generation — requires v1
client_generate = genai.Client(
    api_key=os.getenv('GOOGLE_API_KEY'),
    http_options={'api_version': 'v1'}
)

SYSTEM_PROMPT = """You are an official academic policy assistant for Middlesex University Mauritius (MUM).
Your purpose is to help students and staff understand MUM academic policies accurately and clearly.

Rules you must ALWAYS follow:
1. Answer ONLY using the retrieved policy passages provided in the context below.
2. For EVERY factual claim, cite the source like this: [Source: Document Title]
3. When summarizing, use clear ## headings for each section and bullet points for key details.
4. If the context does not contain enough information, respond with exactly:
   'I cannot find a definitive answer in the MUM policy documents. Please contact the relevant department directly.'
5. NEVER invent, guess, or use information not present in the provided context.
6. Use plain, clear English suitable for university students.
7. For disciplinary or sensitive matters, always remind the student that support is available at careandconcern@mdx.ac.mu"""

# Length instructions for different response lengths.
#  These are included in the prompt to guide the model's verbosity and structure.
LENGTH_INSTRUCTIONS = {
    'Brief': (
        'Be concise. Give only the most essential facts in 2-4 short bullet points. '
        'No sub-headings. No elaboration. Maximum 80 words.'
    ),
    'Standard': (
        'Give a clear, balanced answer. Use bullet points where helpful. '
        'Cover the main points without being exhaustive. Around 150-250 words.'
    ),
    'Detailed': (
        'Give a comprehensive, thorough answer. Use ## headings, ### sub-headings, '
        'and bullet points. Cover all relevant details, exceptions, and examples. '
        '300+ words.'
    ),
}

# Language instructions — injected into the system prompt when a non-English language is selected.
# The source documents are always in English; only the generated response is translated.
LANGUAGE_INSTRUCTIONS = {
    'English': '',
    'French':  'IMPORTANT: You MUST write your entire response in French (Français). Translate all policy information into clear, natural French. Keep source citation names in English (e.g. [Source: Student Conduct Rules]).',
}

def get_embedding(text):
    """Get vector embedding for text using Gemini embedding model."""
    try:
        result = client_embed.models.embed_content(
            model='gemini-embedding-001',
            contents=text
        )
        return result.embeddings[0].values
    except Exception as e:
        logging.error(f'Embedding error: {e}')
        raise

def compute_confidence(dist):
    """Convert cosine distance to a confidence score with power curve."""
    raw     = max(0.0, 1.0 - dist)
    boosted = raw ** 0.65
    return round(min(1.0, boosted), 4)

def extract_year(source_name):
    """Extract a 4-digit year from a filename. Returns the year as int, or None."""
    match = re.search(r'(20\d{2})', source_name)
    return int(match.group(1)) if match else None

def extract_base_name(source_name):
    """
    Strip the year from a filename to get the base document identity.
    e.g. 'Student_Conduct_2024.pdf' → 'Student_Conduct_.pdf'
         'Data_Protection.pdf' → 'Data_Protection.pdf' (unchanged)
    This lets us detect when two files are versions of the same document.
    """
    return re.sub(r'20\d{2}', '', source_name)

def filter_outdated_chunks(chunks):
    """
    When multiple versions of the same document exist (same base name,
    different year), keep ONLY chunks from the most recent version.
    Documents without a year in the filename are always kept.
    """
    # Build a map: base_name → most recent year available
    latest_year = {}
    for c in chunks:
        source = c['meta'].get('source', '')
        year   = extract_year(source)
        if year is None:
            continue
        base = extract_base_name(source)
        if base not in latest_year or year > latest_year[base]:
            latest_year[base] = year

    # Filter: drop chunks from older versions
    filtered = []
    for c in chunks:
        source = c['meta'].get('source', '')
        year   = extract_year(source)
        base   = extract_base_name(source)

        if year is None:
            # No year in filename — always keep
            filtered.append(c)
        elif year >= latest_year.get(base, 0):
            # This is the latest version — keep
            filtered.append(c)
        else:
            # Older version exists with a newer year — drop
            logging.info(f'Filtered outdated chunk from "{source}" (newer version available)')

    return filtered

def retrieve(query, k=TOP_K):
    """Retrieve top-k relevant chunks from ChromaDB using vector similarity."""
    try:
        chroma = chromadb.PersistentClient(path=DB_PATH)
        col    = chroma.get_or_create_collection(COLLECTION)

        count = col.count()
        if count == 0:
            logging.warning('Collection is empty. Run ingest.py first.')
            return [], 0.0

        query_embedding = get_embedding(query)
        results = col.query(
            query_embeddings=[query_embedding],
            n_results=min(k, count),
            include=['documents', 'metadatas', 'distances']
        )

        chunks      = []
        total_score = 0.0

        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            score = compute_confidence(dist)
            total_score += score
            chunks.append({'text': doc, 'meta': meta, 'score': score})

        # Filter out chunks from older versions of the same document.
        # e.g. if both Student_Conduct_2023.pdf and Student_Conduct_2024.pdf
        # are indexed, only chunks from the 2024 version are kept.
        chunks = filter_outdated_chunks(chunks)

        chunks.sort(key=lambda x: x['score'], reverse=True)
        avg_score = total_score / len(chunks) if chunks else 0.0
        return chunks, avg_score

    except Exception as e:
        logging.error(f'Retrieval error: {e}')
        raise

def retrieve_by_source(source_name, k=20):
    """Retrieve all chunks belonging to a specific source document."""
    try:
        chroma  = chromadb.PersistentClient(path=DB_PATH)
        col     = chroma.get_or_create_collection(COLLECTION)
        result  = col.get(include=['documents', 'metadatas'])
        chunks  = []
        for doc, meta in zip(result['documents'], result['metadatas']):
            if meta.get('source') == source_name:
                chunks.append({'text': doc, 'meta': meta, 'score': 1.0})
        return chunks[:k]
    except Exception as e:
        logging.error(f'Retrieve by source error: {e}')
        return []

def build_context(chunks):
    """Format retrieved chunks into a structured context block for the prompt."""
    top_chunks = chunks[:5]
    parts = []
    for i, c in enumerate(top_chunks, 1):
        src       = c['meta'].get('source', 'Unknown')
        doc_title = c['meta'].get('doc_title', src)
        score     = c['score']
        parts.append(
            f'[Reference {i} | Document: {doc_title} | File: {src} | Relevance: {score:.2f}]\n{c["text"]}'
        )
    return '\n\n---\n\n'.join(parts)

def get_sources(chunks):
    """Return a deduplicated list of source metadata dicts for UI display."""
    seen    = set()
    sources = []
    for c in chunks[:6]:
        src       = c['meta'].get('source', 'Unknown')
        doc_title = c['meta'].get('doc_title', src)
        chunk_idx = c['meta'].get('chunk_index', '?')
        key       = f'{src}_{chunk_idx}'
        if key not in seen:
            seen.add(key)
            sources.append({
                'file':    src,
                'title':   doc_title,
                'chunk':   chunk_idx,
                'score':   c['score'],
                'preview': c['text'][:150] + '...' if len(c['text']) > 150 else c['text']
            })
    return sources

def summarise_history(history):
    """
    Compress a long conversation history into a concise summary.
    Called when history exceeds 6 exchanges to keep the prompt within
    token limits while preserving the key topics and decisions.

    Returns a string: the compressed summary + the last 2 verbatim exchanges.
    """
    if not history or len(history) <= 6:
        # Short enough — return as-is in the old format
        parts = ''
        for role, msg in history[-6:]:
            label     = 'Student' if role == 'human' else 'Assistant'
            short_msg = msg[:300] + '...' if len(msg) > 300 else msg
            parts += f'{label}: {short_msg}\n'
        return parts

    # Split: older messages get summarised, recent 4 stay verbatim
    older  = history[:-4]
    recent = history[-4:]

    # Build older transcript for summarisation
    transcript = ''
    for role, msg in older:
        label     = 'Student' if role == 'human' else 'Assistant'
        short_msg = msg[:200] + '...' if len(msg) > 200 else msg
        transcript += f'{label}: {short_msg}\n'

    try:
        prompt = f"""Summarise this conversation between a student and a university policy chatbot
in 3-5 bullet points. Focus on: topics discussed, key facts established,
and any unanswered questions. Be concise — max 120 words total.

Conversation:
{transcript}

Summary:"""

        result = client_generate.models.generate_content(
            model='models/gemini-2.5-flash',
            contents=prompt
        )
        summary = result.text.strip()
        logging.info(f'Compressed {len(older)} older messages into history summary')
    except Exception as e:
        logging.warning(f'History summarisation failed: {e}')
        # Fallback: just use truncated older messages
        summary = transcript[:500]

    # Build final history text: summary + recent verbatim
    recent_text = ''
    for role, msg in recent:
        label     = 'Student' if role == 'human' else 'Assistant'
        short_msg = msg[:300] + '...' if len(msg) > 300 else msg
        recent_text += f'{label}: {short_msg}\n'

    return f'[Summary of earlier conversation]\n{summary}\n\n[Recent exchanges]\n{recent_text}'


def classify_query(query):
    """Detect query intent to apply the most appropriate response format."""
    q = query.lower()
    if any(w in q for w in ['summarize', 'summary', 'overview', 'what is in',
                             'describe the', 'key points', 'tell me about',
                             'what does the', 'outline']):
        return 'summary'
    if any(w in q for w in ['what happens if', 'consequence', 'penalty',
                             'punishment', 'warning point', 'expelled',
                             'suspended', 'dismissed', 'sanction']):
        return 'disciplinary'
    if any(w in q for w in ['how do i', 'how can i', 'steps to', 'process',
                             'procedure', 'apply', 'submit', 'appeal',
                             'what are the steps', 'how to']):
        return 'procedural'
    if any(w in q for w in ['compare', 'difference between', 'versus',
                             ' vs ', 'distinguish', 'contrast']):
        return 'comparison'
    if any(w in q for w in ['define', 'what is', 'what does', 'meaning of',
                             'explain', 'who is', 'what are']):
        return 'definition'
    return 'general'

def answer(query, history=None, length='Standard', language='English'):
    """
    Main RAG pipeline: retrieve -> build context -> generate answer.
    Returns (response_text, sources_list, confidence_score, query_type).
    Accepts length: 'Brief' | 'Standard' | 'Detailed'
    """
    start_time = time.time()

    try:
        chunks, avg_score = retrieve(query)

        if not chunks:
            return (
                '⚠️ The policy document index is empty. '
                'Please run `python ingest.py` first to index your documents.',
                [], 0.0, 'error'
            )

        query_type = classify_query(query)

        if avg_score < MIN_CONFIDENCE and query_type != 'summary':
            logging.info(f'Low confidence ({avg_score:.2f}) for: {query}')
            return (
                'I cannot find a definitive answer in the MUM policy documents. '
                'Please contact the relevant department directly.',
                [], avg_score, query_type
            )

        context      = build_context(chunks)
        sources      = get_sources(chunks)
        history_text = ''

        if history:
            history_text = summarise_history(history)

        type_instructions = {
            'summary':      'Write a comprehensive structured summary. Use ## for main sections, ### for subsections, and bullet points throughout. Cover ALL key areas.',
            'disciplinary': 'List all misconduct categories clearly. For each: show warning point range, specific consequence, and escalation path. Use bullet points and bold key terms.',
            'procedural':   'Number every step (1, 2, 3...). State who is responsible at each stage, any deadlines, and what happens if a step is missed.',
            'comparison':   'Present the comparison in a clear structured format. Use ## headings for each document, then a ## Key Differences section with bullet points.',
            'definition':   'Define the term precisely. Then explain the full policy context, any conditions or exceptions, and give examples from the documents.',
            'general':      'Answer completely and precisely. Cite [Source: Document Title] immediately after every factual statement.',
        }

        length_instruction = LENGTH_INSTRUCTIONS.get(length, LENGTH_INSTRUCTIONS['Standard'])

        # Language instruction — translate response if non-English
        lang_instruction = LANGUAGE_INSTRUCTIONS.get(language, '')

        full_prompt = f"""{SYSTEM_PROMPT}

Query type detected: {query_type}
Formatting instruction: {type_instructions.get(query_type, type_instructions['general'])}
Response length instruction: {length_instruction}
{lang_instruction}

Previous conversation:
{history_text if history_text else 'None'}

Retrieved policy context:
{context}

Student question: {query}

Answer (cite [Source: Document Title] after every fact):"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client_generate.models.generate_content(
                    model='models/gemini-2.5-flash',
                    contents=full_prompt
                )
                break
            except Exception as e:
                if '429' in str(e) and attempt < max_retries - 1:
                    wait_time = 65
                    logging.info(f'Rate limited on attempt {attempt+1}. Waiting {wait_time}s...')
                    time.sleep(wait_time)
                else:
                    raise

        elapsed   = round(time.time() - start_time, 2)
        top_score = chunks[0]['score'] if chunks else 0.0

        logging.info(
            f'Query: "{query[:80]}" | Type: {query_type} | Length: {length} | '
            f'Lang: {language} | AvgScore: {avg_score:.2f} | TopScore: {top_score:.2f} | Time: {elapsed}s'
        )

        return response.text, sources, avg_score, query_type

    except Exception as e:
        logging.error(f'Answer generation error: {e}')
        raise


def compare_documents(doc_a, doc_b, title_a, title_b, length='Standard', language='English'):
    """
    Compare two documents side by side using their full chunk content.
    Returns (response_text, sources_list, confidence_score, query_type).
    """
    start_time = time.time()

    try:
        chunks_a = retrieve_by_source(doc_a)
        chunks_b = retrieve_by_source(doc_b)

        if not chunks_a or not chunks_b:
            missing = title_a if not chunks_a else title_b
            return (
                f'⚠️ Could not find indexed content for **{missing}**. '
                'Please make sure the document is indexed.',
                [], 0.0, 'comparison'
            )

        def build_doc_context(chunks, title, max_chunks=5):
            parts = []
            for i, c in enumerate(chunks[:max_chunks], 1):
                parts.append(f'[{title} — Section {i}]\n{c["text"]}')
            return '\n\n---\n\n'.join(parts)

        context_a = build_doc_context(chunks_a, title_a)
        context_b = build_doc_context(chunks_b, title_b)

        length_instruction = LENGTH_INSTRUCTIONS.get(length, LENGTH_INSTRUCTIONS['Standard'])

        # Language instruction — translate response if non-English
        lang_instruction = LANGUAGE_INSTRUCTIONS.get(language, '')

        prompt = f"""You are an academic policy assistant for Middlesex University Mauritius.
Compare the following two policy documents clearly and objectively.

Structure your comparison exactly like this:
## {title_a}
- Key points, scope, and main provisions

## {title_b}
- Key points, scope, and main provisions

## Key Similarities
- What both documents have in common

## Key Differences
- How they differ in purpose, scope, rules, or consequences

## Summary
One paragraph summarising the most important distinction between the two.

Response length instruction: {length_instruction}
{lang_instruction}
Cite [Source: Document Title] after every fact.
ONLY use information from the provided context below.

--- DOCUMENT A: {title_a} ---
{context_a}

--- DOCUMENT B: {title_b} ---
{context_b}

Provide the structured comparison now:"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client_generate.models.generate_content(
                    model='models/gemini-2.5-flash',
                    contents=prompt
                )
                break
            except Exception as e:
                if '429' in str(e) and attempt < max_retries - 1:
                    time.sleep(65)
                else:
                    raise

        sources = get_sources(chunks_a[:3] + chunks_b[:3])
        elapsed = round(time.time() - start_time, 2)

        logging.info(
            f'Comparison: "{title_a}" vs "{title_b}" | Length: {length} | '
            f'Lang: {language} | Time: {elapsed}s'
        )

        return response.text, sources, 1.0, 'comparison'

    except Exception as e:
        logging.error(f'Comparison error: {e}')
        raise


def detect_coverage_gaps(query, response, sources, language='English'):
    """
    Analyse whether the retrieved context fully addressed the student's question.
    Uses a lightweight Gemini call to identify topics mentioned in the query
    that were NOT adequately covered by the retrieved documents.

    Returns a list of gap strings, or [] if the question was fully covered.
    """
    try:
        # Build a rough context summary from source previews
        source_summary = '\n'.join(
            f'- {s["title"]}: {s["preview"]}' for s in sources[:5]
        ) if sources else 'No sources retrieved.'

        lang_instruction = LANGUAGE_INSTRUCTIONS.get(language, '')

        prompt = f"""You are a quality-checker for a university policy chatbot.
Analyse whether the assistant's answer fully addressed the student's question
using ONLY the retrieved policy context.

Rules:
- If the answer fully covers the question, respond with exactly: FULLY_COVERED
- If there are topics the student asked about that the retrieved documents did NOT contain,
  list each missing topic on a new line, prefixed with "GAP: "
- Only flag genuine gaps — do not flag stylistic preferences or minor omissions.
- Maximum 3 gaps.
{lang_instruction}

Student's question: {query}

Available source documents:
{source_summary}

Assistant's answer (excerpt): {response[:600]}

Analysis:"""

        result = client_generate.models.generate_content(
            model='models/gemini-2.5-flash',
            contents=prompt
        )

        text = result.text.strip()

        if 'FULLY_COVERED' in text:
            return []

        gaps = []
        for line in text.splitlines():
            line = line.strip()
            if line.upper().startswith('GAP:'):
                gap = line[4:].strip()
                if len(gap) > 5:
                    gaps.append(gap)

        if gaps:
            logging.info(f'Coverage gaps detected for "{query[:60]}": {gaps}')
        return gaps[:3]

    except Exception as e:
        logging.warning(f'Coverage gap detection failed: {e}')
        return []


def generate_followups(query, response, history=None, language='English'):
    """
    Generate context-aware follow-up suggestions using a dedicated lightweight
    Gemini call. Analyses the current Q&A exchange plus conversation history
    to suggest questions that explore NEW angles not yet covered.

    Returns a list of 3 follow-up question strings, or [] on failure.
    """
    try:
        # Build a summary of questions already asked in the session
        asked_summary = ''
        if history:
            asked_qs = [msg for role, msg in history[-10:] if role == 'human']
            if asked_qs:
                asked_summary = 'Questions already asked in this session:\n' + '\n'.join(f'- {q}' for q in asked_qs)

        lang_instruction = LANGUAGE_INSTRUCTIONS.get(language, '')

        prompt = f"""You are a follow-up question generator for a university policy chatbot.
Based on the Q&A exchange below, suggest exactly 3 follow-up questions the student would naturally want to ask next.

Rules:
- Each question must explore a NEW angle, detail, or related topic NOT already covered in the answer or conversation history.
- Questions must be specific and answerable from MUM policy documents (e.g. misconduct, appeals, data protection, deferrals, safeguarding).
- Keep each question concise — under 15 words.
- Return ONLY the 3 questions, one per line, numbered 1. 2. 3.
- No preamble, no explanation, no extra text.
{lang_instruction}

Student's question: {query}

Assistant's answer (excerpt): {response[:600]}

{asked_summary}

3 follow-up questions:"""

        result = client_generate.models.generate_content(
            model='models/gemini-2.5-flash',
            contents=prompt
        )

        # Parse numbered questions from the response
        questions = []
        for line in result.text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            # Strip numbering like "1.", "1)", "- ", "* "
            line = re.sub(r'^[\d\.\)\-\*]+\s*', '', line).strip()
            line = line.strip('"\'')
            if len(line) > 10:
                questions.append(line)

        logging.info(f'Generated {len(questions[:3])} follow-up suggestions for: "{query[:60]}"')
        return questions[:3]

    except Exception as e:
        logging.warning(f'Follow-up generation failed: {e}')
        return []


if __name__ == '__main__':
    test_queries = [
        'What is the minimum notice given before a disciplinary panel?',
        'How many warning points lead to expulsion?',
        'Summarize the student conduct document',
        'What is the appeal process?',
        'What are the types of misconduct?',
    ]
    for q in test_queries:
        print(f'\nQ: {q}')
        try:
            resp, srcs, score, qtype = answer(q, length='Standard')
            print(f'Type: {qtype} | Score: {score:.2f}')
            print(f'A: {resp[:400]}')
            print(f'Sources: {[s["file"] for s in srcs]}')

            followups = generate_followups(q, resp)
            print(f'Follow-ups: {followups}')
        except Exception as e:
            print(f'ERROR: {e}')
        time.sleep(2)