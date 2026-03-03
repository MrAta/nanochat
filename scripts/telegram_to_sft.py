"""
Convert a raw Telegram channel export (data/telegram_raw.jsonl) into SFT
conversation data (data/telegram_sft.jsonl) ready for the CustomJSON task loader.

Two modes:
  llm      — call OpenRouter to synthesize a plausible user question for each post
  template — use keyword-heuristic topic extraction + canned question templates (free)

Reply threading: if a post replies to another post in the same export, the
replied-to post becomes the user turn and the reply becomes the assistant turn.

Usage:
    # LLM mode (recommended, needs OPENROUTER_API_KEY)
    python -m scripts.telegram_to_sft \
        --input data/telegram_raw.jsonl \
        --output data/telegram_sft.jsonl \
        --mode llm \
        --domain-description "a Persian political analyst commenting on Iranian social and political affairs" \
        --workers 8

    # Template mode (free, no API key needed)
    python -m scripts.telegram_to_sft \
        --input data/telegram_raw.jsonl \
        --output data/telegram_sft.jsonl \
        --mode template
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Telegram raw export to SFT JSONL"
    )
    parser.add_argument("--input", type=str, default="data/telegram_raw.jsonl",
                        help="Input JSONL file from telegram_export.py")
    parser.add_argument("--output", type=str, default="data/telegram_sft.jsonl",
                        help="Output SFT JSONL file")
    parser.add_argument("--mode", type=str, default="llm",
                        choices=["llm", "template"],
                        help="User-prompt generation mode (default: llm)")
    parser.add_argument("--domain-description", type=str,
                        default="a subject-matter expert commenting on current affairs",
                        help="One-line description of the channel author's expertise/domain")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for LLM API calls (default: 4)")
    parser.add_argument("--min-len", type=int, default=30,
                        help="Skip posts shorter than this (default: 30)")
    parser.add_argument("--strip-hashtags", action="store_true", default=True,
                        help="Strip #hashtags from post text (default: True)")
    parser.add_argument("--no-strip-hashtags", dest="strip_hashtags",
                        action="store_false")
    parser.add_argument("--strip-mentions", action="store_true", default=True,
                        help="Strip @mentions from post text (default: True)")
    parser.add_argument("--no-strip-mentions", dest="strip_mentions",
                        action="store_false")
    parser.add_argument("--model", type=str,
                        default="google/gemini-2-flash-lite",
                        help="OpenRouter model for LLM mode (default: google/gemini-2-flash-lite)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str, strip_hashtags: bool, strip_mentions: bool) -> str:
    if strip_hashtags:
        text = re.sub(r"#\S+", "", text)
    if strip_mentions:
        text = re.sub(r"@\S+", "", text)
    # Collapse multiple whitespace / newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Template mode
# ---------------------------------------------------------------------------

TOPIC_KEYWORDS = {
    "politics": ["سیاس", "دولت", "مجلس", "رئیس", "انتخاب",
                 "politic", "govern", "parliament", "election", "president"],
    "economy": ["اقتصاد", "تورم", "بازار", "ارز", "نفت",
                 "econom", "inflation", "market", "currency", "oil"],
    "society": ["جامع", "مردم", "اعتراض", "حقوق",
                 "society", "people", "protest", "rights"],
    "international": ["بین‌الملل", "آمریک", "اروپ", "روسی", "چین",
                      "international", "america", "europe", "russia", "china"],
    "culture": ["فرهنگ", "هنر", "ادبیات", "موسیقی",
                 "culture", "art", "literature", "music"],
}

TEMPLATES_BY_TOPIC = {
    "politics": [
        "What is your analysis of the current political situation?",
        "How do you view the government's recent decisions?",
        "What are the key political dynamics at play here?",
    ],
    "economy": [
        "What is your take on the economic situation?",
        "How do you assess recent economic developments?",
        "What does this mean for the country's economy?",
    ],
    "society": [
        "What is your perspective on recent social developments?",
        "How do you view the situation unfolding in society?",
        "What does this reveal about social conditions?",
    ],
    "international": [
        "How do you analyze the international dimension of this?",
        "What is your view on the foreign policy implications?",
        "How does this relate to regional and global dynamics?",
    ],
    "culture": [
        "What is your take on the cultural significance of this?",
        "How do you view recent cultural developments?",
    ],
    "default": [
        "What is your view on this?",
        "What do you make of the current situation?",
        "How would you analyze what is happening?",
        "What is your perspective on recent events?",
        "Can you share your thoughts on this development?",
    ],
}


def infer_topic(text: str) -> str:
    text_lower = text.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return topic
    return "default"


def template_question(text: str) -> str:
    import random
    topic = infer_topic(text)
    templates = TEMPLATES_BY_TOPIC.get(topic, TEMPLATES_BY_TOPIC["default"])
    return random.choice(templates)


# ---------------------------------------------------------------------------
# LLM mode
# ---------------------------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

QUESTION_GEN_PROMPT = """\
You are helping build a dataset to fine-tune a language model that will learn the voice and perspective of {domain_description}.

Below is a post written by this person:

---
{post_text}
---

Your task: write a single, natural user question (in English) that someone might ask which would prompt exactly this post as the answer. The question should:
- Be a genuine question a curious person would ask
- Be specific enough to elicit this particular response
- Sound natural, not robotic
- Be 1-2 sentences maximum

Respond with only the question, nothing else."""


def llm_question(text: str, domain_description: str, model: str,
                 api_key: str) -> str:
    import requests

    prompt = QUESTION_GEN_PROMPT.format(
        domain_description=domain_description,
        post_text=text,
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "stream": False,
        "temperature": 0.9,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    result = response.json()
    if "error" in result:
        raise RuntimeError(f"OpenRouter API error: {result['error']}")
    return result["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Reply threading
# ---------------------------------------------------------------------------

def build_reply_pairs(posts: list[dict]) -> list[dict]:
    """
    For posts that are replies to another post in the same export, build
    2-turn conversations: [user: replied-to text, assistant: reply text].
    Returns list of SFT conversation dicts.
    """
    id_to_post = {p["id"]: p for p in posts}
    pairs = []
    for post in posts:
        reply_id = post.get("reply_to_msg_id")
        if reply_id and reply_id in id_to_post:
            parent = id_to_post[reply_id]
            parent_text = parent.get("_cleaned_text", parent["text"])
            reply_text = post.get("_cleaned_text", post["text"])
            if parent_text and reply_text:
                pairs.append([
                    {"role": "user", "content": parent_text},
                    {"role": "assistant", "content": reply_text},
                ])
    return pairs


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_conversation(messages: list[dict]) -> bool:
    if len(messages) < 2:
        return False
    for i, msg in enumerate(messages):
        if "role" not in msg or "content" not in msg:
            return False
        expected = "user" if i % 2 == 0 else "assistant"
        if msg["role"] != expected:
            return False
        if not msg["content"].strip():
            return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load raw posts
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    posts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            posts.append(json.loads(line))

    print(f"Loaded {len(posts)} posts from {args.input}")

    # Clean texts
    for post in posts:
        post["_cleaned_text"] = clean_text(
            post["text"],
            strip_hashtags=args.strip_hashtags,
            strip_mentions=args.strip_mentions,
        )

    # Filter short posts
    filtered_posts = [p for p in posts if len(p["_cleaned_text"]) >= args.min_len]
    print(f"After min-len filter ({args.min_len} chars): {len(filtered_posts)} posts")

    # Build reply-threaded pairs (natural multi-turn signal)
    reply_conversations = build_reply_pairs(posts)
    print(f"Built {len(reply_conversations)} reply-threaded pairs")

    # Build single-post conversations using user prompt generation
    api_key = None
    if args.mode == "llm":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("ERROR: OPENROUTER_API_KEY not set. Use --mode template or set the key.")
            sys.exit(1)

    # Collect single-post conversations
    single_conversations = []

    if args.mode == "template":
        for post in filtered_posts:
            text = post["_cleaned_text"]
            question = template_question(text)
            single_conversations.append([
                {"role": "user", "content": question},
                {"role": "assistant", "content": text},
            ])
        print(f"Generated {len(single_conversations)} template conversations")

    else:  # llm mode
        print(f"Generating questions via LLM ({args.model}) with {args.workers} workers...")
        completed = 0
        errors = 0

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_post = {
                executor.submit(
                    llm_question,
                    post["_cleaned_text"],
                    args.domain_description,
                    args.model,
                    api_key,
                ): post
                for post in filtered_posts
            }
            for future in as_completed(future_to_post):
                post = future_to_post[future]
                try:
                    question = future.result()
                    single_conversations.append([
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": post["_cleaned_text"]},
                    ])
                    completed += 1
                    if completed % 50 == 0:
                        print(f"  {completed}/{len(filtered_posts)} LLM questions generated...")
                except Exception as e:
                    errors += 1
                    print(f"  [ERROR] post id={post['id']}: {e}")

        print(f"LLM mode: {completed} generated, {errors} errors")

    # Merge all conversations
    all_conversations = reply_conversations + single_conversations

    # Validate and write
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    written = 0
    skipped = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for conv in all_conversations:
            if not validate_conversation(conv):
                skipped += 1
                continue
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nDone. Wrote {written} conversations to {args.output} "
          f"({skipped} skipped due to validation errors)")
    print(f"Validate with: CustomJSON(filepath='{args.output}')")


if __name__ == "__main__":
    main()
