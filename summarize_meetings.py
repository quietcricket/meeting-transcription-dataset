#!/usr/bin/env python3
"""
Meeting transcript analyzer.

Pipeline:
  1. NLTK – trim transcripts (remove filler-only turns, very short segments,
             deduplicate back-to-back identical speakers)
  2. LLM  – analyze trimmed transcript for summary, participant views,
             agreements, disagreements, and most time-consuming topic.

Output: meeting_summaries.json
"""
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import nltk
from dotenv import load_dotenv

load_dotenv()

# ── NLTK setup ────────────────────────────────────────────────────────────────
for resource in ("stopwords", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource.startswith("punkt") else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords as _sw

STOPWORDS = set(_sw.words("english"))

FILLER_WORDS = {
    "um", "uh", "hmm", "mm", "ah", "oh", "er", "uhh", "umm", "hm",
    "yeah", "yep", "yup", "okay", "ok", "right", "sure", "fine",
    "yes", "no", "nope", "hi", "hello", "bye", "thanks", "thank",
    "good", "great", "nice", "wow", "well", "like", "so", "just",
    "actually", "basically", "literally", "honestly", "seriously",
}

FILLER_PATTERN = re.compile(r"^[\W\s]*$|^[^a-zA-Z]*$")

# ── LLM client ────────────────────────────────────────────────────────────────
API_KEY  = os.environ["API_KEY"]
BASE_URL = os.environ["BASE_URL"]
MODEL    = os.environ["MODEL"]

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=300.0)

# ── NLTK preprocessing ───────────────────────────────────────────────────────

def is_filler_turn(text: str) -> bool:
    """Return True if the turn contains no meaningful content."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return True
    meaningful = [w for w in words if w not in FILLER_WORDS and w not in STOPWORDS]
    # Keep turn if at least 3 meaningful words or >30% of words are meaningful
    return len(meaningful) < 3 and (len(meaningful) / len(words)) < 0.3


def trim_transcript(segments: list) -> list:
    """
    Use NLTK / heuristics to:
      - Remove filler-only turns
      - Merge consecutive segments from the same speaker
      - Drop turns shorter than 4 words that carry no information
    Returns list of {speaker, text} dicts.
    """
    # Merge consecutive same-speaker segments
    merged = []
    for seg in segments:
        speaker = seg.get("speakerName", "Unknown").strip()
        text    = seg.get("text", "").strip()
        if not text:
            continue
        if merged and merged[-1]["speaker"] == speaker:
            merged[-1]["text"] += " " + text
        else:
            merged.append({"speaker": speaker, "text": text})

    # Filter filler turns
    filtered = []
    for turn in merged:
        words = turn["text"].split()
        if len(words) < 4:
            continue
        if is_filler_turn(turn["text"]):
            continue
        filtered.append(turn)

    return filtered


MAX_TURNS = 120  # cap to avoid context overflow on very large meetings

def build_transcript_text(turns: list) -> str:
    """Build transcript string, sampling evenly if over MAX_TURNS."""
    if len(turns) > MAX_TURNS:
        head_n = MAX_TURNS // 10
        tail_n = MAX_TURNS // 10
        mid_n  = MAX_TURNS - head_n - tail_n
        head   = turns[:head_n]
        tail   = turns[-tail_n:]
        mid    = turns[head_n:-tail_n]
        step   = max(1, len(mid) // mid_n)
        mid_sample = mid[::step][:mid_n]
        sampled = head + mid_sample + tail
        note = f"[NOTE: Transcript sampled from {len(turns)} to {len(sampled)} turns for analysis]\n"
    else:
        sampled = turns
        note = ""
    lines = [f"{t['speaker']}: {t['text']}" for t in sampled]
    return note + "\n".join(lines)


# ── LLM analysis ──────────────────────────────────────────────────────────────

ANALYSIS_PROMPT_TEMPLATE = """Analyze the following meeting transcript and return ONLY a JSON object with these exact keys.
Do not include any explanation, markdown, or text outside the JSON.

Required JSON structure:
{{
  "summary": "2-4 sentence description of what the meeting was about and its main purpose",
  "participant_views": {{
    "<speaker name>": ["concise statement of a distinct view/position/suggestion they expressed"]
  }},
  "agreements": [
    {{"topic": "what was agreed", "details": "brief description", "participants": ["names"]}}
  ],
  "disagreements": [
    {{"topic": "what was disputed", "details": "brief description", "participants": ["names"]}}
  ],
  "most_time_consuming": {{
    "topic": "the subject that consumed the most discussion time",
    "reason": "why it took so long"
  }}
}}

Rules:
- Return ONLY valid JSON.
- participant_views: meaningful positions only (skip backchannels). Up to 5 per participant.
- agreements / disagreements: up to 8 each.
- Empty list/object if nothing found.

Meeting ID: {meeting_id}

TRANSCRIPT:
{transcript}"""


def extract_json(text: str) -> str:
    """Extract JSON object from text that may contain markdown fences or preamble."""
    if not text:
        return ""
    # Try to find a JSON object directly
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def analyze_with_llm(meeting_id: str, turns: list, retries: int = 3) -> dict:
    transcript = build_transcript_text(turns)
    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        meeting_id=meeting_id,
        transcript=transcript,
    )

    for attempt in range(retries):
        raw = ""
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=4096,
            )
            raw = (response.choices[0].message.content or "").strip()
            if not raw:
                # Some reasoning models put output only in reasoning_content
                raw = (response.choices[0].message.reasoning_content or "").strip()
            candidate = extract_json(raw)
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            print(f"  [!] JSON parse error on attempt {attempt+1}: {e} | raw[:100]={raw[:100]!r}")
            if attempt == retries - 1:
                return {"error": "JSON parse failed", "raw": raw[:500]}
            time.sleep(2)
        except openai.RateLimitError:
            wait = 2 ** attempt * 5
            print(f"  [!] Rate limit, waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"  [!] API error on attempt {attempt+1}: {e}")
            if attempt == retries - 1:
                return {"error": str(e)}
            time.sleep(3)

    return {"error": "All retries failed"}


# ── Per-participant stats (NLTK) ───────────────────────────────────────────────

def participant_stats(turns: list) -> dict:
    stats = defaultdict(lambda: {"turn_count": 0, "word_count": 0})
    for t in turns:
        stats[t["speaker"]]["turn_count"] += 1
        stats[t["speaker"]]["word_count"] += len(t["text"].split())
    return dict(stats)


# ── Main pipeline ────────────────────────────────────────────────────────────

def process_meeting(record: dict) -> dict:
    meeting    = record.get("meeting", {})
    meeting_id = meeting.get("meetingId", record.get("dialogId", "unknown"))
    dialog_id  = record.get("dialogId", "unknown")
    segments   = meeting.get("transcriptSegments", [])

    # Step 1: NLTK trim
    turns       = trim_transcript(segments)
    participants = sorted({t["speaker"] for t in turns})

    # Step 2: LLM analysis
    llm_result  = analyze_with_llm(meeting_id, turns)

    return {
        "dialogId":            dialog_id,
        "meetingId":           meeting_id,
        "participants":        participants,
        "participant_stats":   participant_stats(turns),
        "summary":             llm_result.get("summary", ""),
        "participant_views":   llm_result.get("participant_views", {}),
        "agreements":          llm_result.get("agreements", []),
        "disagreements":       llm_result.get("disagreements", []),
        "most_time_consuming": llm_result.get("most_time_consuming", {}),
        "_llm_error":          llm_result.get("error"),  # None if success
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze meeting transcripts with NLTK + LLM")
    parser.add_argument("input",  help="Input JSONL file (e.g. train.jsonl)")
    parser.add_argument("output", help="Output JSON file (e.g. train_summaries.json)")
    args = parser.parse_args()

    input_path   = args.input
    output_path  = args.output
    cache_path   = output_path.replace(".json", ".cache.jsonl")

    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed line: {e}")

    # Load already-processed results from cache (resume support)
    done_keys: set = set()
    cached: list   = []
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    key = (r.get("dialogId",""), r.get("meetingId",""))
                    done_keys.add(key)
                    cached.append(r)
        print(f"Resuming: {len(cached)} already done, {len(records)-len(cached)} remaining\n")
    else:
        print(f"Processing {len(records)} meetings...\n")

    cache_fh = open(cache_path, "a")

    def process_one(args):
        i, record = args
        meeting_id = record.get("meeting", {}).get("meetingId", "?")
        dialog_id  = record.get("dialogId", "?")
        key = (dialog_id, meeting_id)
        if key in done_keys:
            return None  # already done
        try:
            result = process_meeting(record)
            return (i, meeting_id, result)
        except Exception as e:
            return (i, meeting_id, {"error": str(e)})

    results = list(cached)
    total   = len(records)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(process_one, (i, r)): i for i, r in enumerate(records)}
        for future in as_completed(futures):
            ret = future.result()
            if ret is None:
                continue
            i, meeting_id, result = ret
            err = result.get("error") or result.get("_llm_error")
            status = "OK" if not err else f"WARN: {err}"
            print(f"[{i+1:2d}/{total}] {meeting_id} ... {status}", flush=True)
            results.append(result)
            # Write to cache immediately
            cache_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            cache_fh.flush()

    cache_fh.close()

    # Clean up _llm_error from successful entries and write final output
    for r in results:
        if r.get("_llm_error") is None:
            r.pop("_llm_error", None)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    ok   = sum(1 for r in results if "error" not in r and "_llm_error" not in r)
    errs = len(results) - ok
    print(f"\nDone. {ok} succeeded, {errs} errors → {output_path}")
    if ok == total:
        os.remove(cache_path)
        print("Cache cleaned up.")


if __name__ == "__main__":
    main()
