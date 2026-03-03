"""
Export all text messages from a Telegram channel to JSONL.

Authenticates interactively via phone + OTP (MTProto), saves session to
.telegram_session so subsequent runs skip auth.

Usage:
    python -m scripts.telegram_export \
        --channel @channelusername \
        --output data/telegram_raw.jsonl \
        --min-len 30

Required env vars (or .env file):
    TELEGRAM_API_ID     — from https://my.telegram.org/apps
    TELEGRAM_API_HASH   — from https://my.telegram.org/apps

Optional env vars:
    TELEGRAM_CHANNEL    — channel username/invite (can also use --channel flag)
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Export a Telegram channel to JSONL")
    parser.add_argument("--channel", type=str, default=None,
                        help="Channel username or invite link (e.g. @channelusername). "
                             "Defaults to TELEGRAM_CHANNEL env var.")
    parser.add_argument("--output", type=str, default="data/telegram_raw.jsonl",
                        help="Output JSONL file path (default: data/telegram_raw.jsonl)")
    parser.add_argument("--min-len", type=int, default=30,
                        help="Skip messages shorter than this many characters (default: 30)")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Number of messages per API request (default: 100, max: 100)")
    parser.add_argument("--session", type=str, default=".telegram_session",
                        help="Path to session file (default: .telegram_session)")
    return parser.parse_args()


async def export_channel(args):
    try:
        from telethon import TelegramClient
        from telethon.tl.types import MessageFwdHeader
    except ImportError:
        print("ERROR: telethon is not installed. Run: pip install telethon")
        sys.exit(1)

    api_id = os.environ.get("TELEGRAM_API_ID")
    api_hash = os.environ.get("TELEGRAM_API_HASH")
    if not api_id or not api_hash:
        print("ERROR: TELEGRAM_API_ID and TELEGRAM_API_HASH must be set "
              "(env vars or .env file).")
        sys.exit(1)
    api_id = int(api_id)

    channel = args.channel or os.environ.get("TELEGRAM_CHANNEL")
    if not channel:
        print("ERROR: Provide --channel or set TELEGRAM_CHANNEL env var.")
        sys.exit(1)

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    client = TelegramClient(args.session, api_id, api_hash)
    await client.start()  # interactive phone + OTP if no session saved

    print(f"Connected. Fetching messages from: {channel}")

    total_fetched = 0
    total_written = 0

    with open(args.output, "w", encoding="utf-8") as f:
        async for message in client.iter_messages(channel, limit=None):
            total_fetched += 1

            # Skip media-only posts (no text)
            if not message.text:
                continue

            text = message.text.strip()
            if len(text) < args.min_len:
                continue

            record = {
                "id": message.id,
                "date": message.date.isoformat() if message.date else None,
                "text": text,
                "reply_to_msg_id": message.reply_to_msg_id if message.reply_to else None,
                "views": message.views,
                "fwd_from": bool(message.fwd_from),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_written += 1

            if total_written % 500 == 0:
                print(f"  Written {total_written} messages "
                      f"(fetched {total_fetched} total)...")

    await client.disconnect()
    print(f"\nDone. Fetched {total_fetched} messages, wrote {total_written} to {args.output}")


def main():
    args = parse_args()
    asyncio.run(export_channel(args))


if __name__ == "__main__":
    main()
