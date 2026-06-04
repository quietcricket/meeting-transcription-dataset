#!/usr/bin/env python3
"""
AliMeeting transcript analyzer.

Pipeline:
  1. Parse Praat TextGrid files → chronologically sorted speaker turns
  2. NLTK-style trim (filler removal, merge consecutive same-speaker turns,
     drop very short / filler-only segments) — Chinese-aware
  3. LLM (gpt-oss-120b) → structured JSON summary

Usage:
    python3 summarize_alimeeting.py <input_dir> <output.json>

Example:
    python3 summarize_alimeeting.py alimeeting-eval alimeeting_summaries.json
"""
import json
import os
import re
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
from dotenv import load_dotenv

load_dotenv()

# ── LLM client ────────────────────────────────────────────────────────────────
API_KEY  = os.environ["API_KEY"]
BASE_URL = os.environ["BASE_URL"]
MODEL    = os.environ["MODEL"]

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=300.0)

# ── Chinese filler characters / words ─────────────────────────────────────────
# Common Mandarin backchannel / filler expressions
CHINESE_FILLERS = {
    "嗯", "啊", "哦", "哈", "呢", "啦", "吧", "哎", "喂", "唉",
    "嗯嗯", "哦哦", "啊啊", "好好", "对对", "是是", "嗯啊",
    "好", "对", "是", "嗯哦", "哎哎",
}

def is_filler_turn_zh(text: str) -> bool:
    """Return True if the turn is filler-only (Chinese-aware)."""
    stripped = text.strip()
    if not stripped:
        return True
    # Remove punctuation
    clean = re.sub(r"[，。！？、；：\u201c\u201d\u2018\u2019「」【】（）《》\s]", "", stripped)
    if not clean:
        return True
    # If entire text is in the filler set, skip
    if clean in CHINESE_FILLERS:
        return True
    # Very short (≤2 CJK chars) and matches filler pattern
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", clean)
    if len(cjk_chars) <= 2 and clean in CHINESE_FILLERS:
        return True
    # All characters are repeated filler tokens
    tokens = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+", clean)
    if all(t in CHINESE_FILLERS for t in tokens):
        return True
    return False

def char_count(text: str) -> int:
    """Count meaningful CJK characters + latin words."""
    return len(re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text))

# ── TextGrid parser ────────────────────────────────────────────────────────────

def parse_textgrid(path: str) -> list:
    """
    Parse a Praat TextGrid file and return a list of
    {speaker, text, xmin, xmax} dicts sorted by start time.
    Empty intervals are dropped.
    """
    with open(path, encoding="utf-8") as f:
        content = f.read()

    utterances = []

    # Split into tier blocks
    tier_blocks = re.split(r'\s+item\s*\[\d+\]:', content)
    for block in tier_blocks[1:]:  # skip header
        # Extract speaker name
        name_match = re.search(r'name\s*=\s*"([^"]+)"', block)
        if not name_match:
            continue
        speaker = name_match.group(1).strip()

        # Extract all intervals
        interval_blocks = re.split(r'\s+intervals\s*\[\d+\]:', block)
        for ib in interval_blocks[1:]:
            xmin_m = re.search(r'xmin\s*=\s*([\d.]+)', ib)
            xmax_m = re.search(r'xmax\s*=\s*([\d.]+)', ib)
            text_m = re.search(r'text\s*=\s*"((?:[^"\\]|\\.)*)"', ib, re.DOTALL)
            if not (xmin_m and xmax_m and text_m):
                continue
            text = text_m.group(1).strip()
            if not text:
                continue
            utterances.append({
                "speaker": speaker,
                "text":    text,
                "xmin":    float(xmin_m.group(1)),
                "xmax":    float(xmax_m.group(1)),
            })

    # Sort chronologically
    utterances.sort(key=lambda u: u["xmin"])
    return utterances


# ── Transcript trimming ────────────────────────────────────────────────────────

def trim_transcript_zh(utterances: list) -> list:
    """
    1. Merge consecutive same-speaker turns (within 2s gap)
    2. Drop filler-only turns
    3. Drop turns with < 4 meaningful characters
    """
    if not utterances:
        return []

    # Merge consecutive same-speaker segments (gap ≤ 2s)
    merged = []
    for u in utterances:
        if (merged
                and merged[-1]["speaker"] == u["speaker"]
                and u["xmin"] - merged[-1]["xmax"] <= 2.0):
            merged[-1]["text"] += u["text"]
            merged[-1]["xmax"] = u["xmax"]
        else:
            merged.append(dict(u))

    # Filter
    filtered = []
    for turn in merged:
        if char_count(turn["text"]) < 4:
            continue
        if is_filler_turn_zh(turn["text"]):
            continue
        filtered.append({"speaker": turn["speaker"], "text": turn["text"]})

    return filtered


# ── Transcript builder (cap large meetings) ───────────────────────────────────

MAX_TURNS = 120

def build_transcript_text(turns: list) -> str:
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
        note = f"[NOTE: Transcript sampled from {len(turns)} to {len(sampled)} turns]\n"
    else:
        sampled = turns
        note = ""
    lines = [f"{t['speaker']}: {t['text']}" for t in sampled]
    return note + "\n".join(lines)


# ── LLM analysis ──────────────────────────────────────────────────────────────

ANALYSIS_PROMPT_TEMPLATE = """Analyze the following meeting transcript (in Chinese) and return ONLY a JSON object with these exact keys.
Do not include any explanation, markdown, or text outside the JSON.
Write all values in English.

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
- Write all values in English.
- participant_views: meaningful positions only (skip backchannels). Up to 5 per participant.
- agreements / disagreements: up to 8 each.
- Empty list/object if nothing found.

Meeting ID: {meeting_id}

TRANSCRIPT:
{transcript}"""


def extract_json(text: str) -> str:
    if not text:
        return ""
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def analyze_with_llm(meeting_id: str, turns: list, retries: int = 5) -> dict:
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
                raw = (getattr(response.choices[0].message, "reasoning_content", None) or "").strip()
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
        except openai.APIStatusError as e:
            if e.status_code == 503:
                wait = 15 * (attempt + 1)
                print(f"  [!] 503 model loading on attempt {attempt+1}, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [!] API status error on attempt {attempt+1}: {e}")
                if attempt == retries - 1:
                    return {"error": str(e)}
                time.sleep(5)
        except Exception as e:
            print(f"  [!] API error on attempt {attempt+1}: {e}")
            if attempt == retries - 1:
                return {"error": str(e)}
            time.sleep(3)

    return {"error": "All retries failed"}


# ── Per-participant stats ──────────────────────────────────────────────────────

def participant_stats(turns: list) -> dict:
    stats = defaultdict(lambda: {"turn_count": 0, "char_count": 0})
    for t in turns:
        stats[t["speaker"]]["turn_count"] += 1
        stats[t["speaker"]]["char_count"] += char_count(t["text"])
    return dict(stats)


# ── Per-meeting pipeline ──────────────────────────────────────────────────────

def process_meeting_textgrid(filepath: str) -> dict:
    meeting_id = Path(filepath).stem  # e.g. R8001_M8004

    utterances   = parse_textgrid(filepath)
    turns        = trim_transcript_zh(utterances)
    participants = sorted({t["speaker"] for t in turns})

    llm_result = analyze_with_llm(meeting_id, turns)

    return {
        "meetingId":           meeting_id,
        "participants":        participants,
        "participant_stats":   participant_stats(turns),
        "summary":             llm_result.get("summary", ""),
        "participant_views":   llm_result.get("participant_views", {}),
        "agreements":          llm_result.get("agreements", []),
        "disagreements":       llm_result.get("disagreements", []),
        "most_time_consuming": llm_result.get("most_time_consuming", {}),
        "_llm_error":          llm_result.get("error"),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze AliMeeting TextGrid transcripts with NLTK + LLM")
    parser.add_argument("input",  help="Input directory containing .TextGrid files")
    parser.add_argument("output", help="Output JSON file (e.g. alimeeting_summaries.json)")
    args = parser.parse_args()

    input_dir   = args.input
    output_path = args.output
    cache_path  = output_path.replace(".json", ".cache.jsonl")

    files = sorted(Path(input_dir).glob("*.TextGrid"))
    if not files:
        print(f"No .TextGrid files found in {input_dir}")
        sys.exit(1)

    # Load cache (resume support)
    done_meeting_ids: set = set()
    cached: list          = []
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done_meeting_ids.add(r.get("meetingId", ""))
                    cached.append(r)
        print(f"Resuming: {len(cached)} already done, {len(files)-len(cached)} remaining\n")
    else:
        print(f"Processing {len(files)} meetings...\n")

    cache_fh   = open(cache_path, "a")
    dedup_lock = threading.Lock()

    def process_one(args):
        i, fpath = args
        meeting_id = fpath.stem
        with dedup_lock:
            if meeting_id in done_meeting_ids:
                return None
            done_meeting_ids.add(meeting_id)
        try:
            result = process_meeting_textgrid(str(fpath))
            return (i, meeting_id, result)
        except Exception as e:
            return (i, meeting_id, {"meetingId": meeting_id, "error": str(e)})

    results = list(cached)
    total   = len(files)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(process_one, (i, f)): i for i, f in enumerate(files)}
        for future in as_completed(futures):
            ret = future.result()
            if ret is None:
                continue
            i, meeting_id, result = ret
            err = result.get("error") or result.get("_llm_error")
            status = "OK" if not err else f"WARN: {str(err)[:60]}"
            print(f"[{i+1:2d}/{total}] {meeting_id} ... {status}", flush=True)
            results.append(result)
            cache_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            cache_fh.flush()

    cache_fh.close()

    # Strip _llm_error from successful entries
    for r in results:
        if r.get("_llm_error") is None:
            r.pop("_llm_error", None)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    ok   = sum(1 for r in results if "error" not in r and "_llm_error" not in r)
    errs = len(results) - ok
    print(f"\nDone. {ok} succeeded, {errs} errors → {output_path}")
    if errs == 0 and os.path.exists(cache_path):
        os.remove(cache_path)
        print("Cache cleaned up.")


if __name__ == "__main__":
    main()
