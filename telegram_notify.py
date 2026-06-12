#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent


def load_env(path: Path = ROOT / ".env") -> None:
    load_dotenv(path, override=False)


def api_call(method: str, payload: dict[str, str] | None = None) -> dict:
    load_env()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured in .env")

    url = f"https://api.telegram.org/bot{token}/{method}"
    data = urlencode(payload or {}).encode()
    request = Request(url, data=data, method="POST")

    try:
        with urlopen(request, timeout=15) as response:
            result = json.loads(response.read().decode())
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Telegram API request failed: {exc}") from exc

    if not result.get("ok"):
        raise RuntimeError(f"Telegram API returned an error: {result}")
    return result


def send(message: str, *, strict: bool = False) -> bool:
    load_env()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

    if not token or not chat_id:
        if strict:
            raise RuntimeError(
                "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be configured in .env"
            )
        print("Telegram notification skipped because credentials are not configured.")
        return False

    try:
        api_call(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": message[:4096],
            },
        )
        return True
    except Exception as exc:
        if strict:
            raise
        print(f"WARNING: Telegram notification failed: {exc}", file=sys.stderr)
        return False


def get_chat_ids() -> list[dict[str, str]]:
    result = api_call("getUpdates", {"limit": "100"})
    found: dict[str, dict[str, str]] = {}

    for update in result.get("result", []):
        message = (
            update.get("message")
            or update.get("edited_message")
            or update.get("channel_post")
        )
        if not message:
            continue
        chat = message.get("chat", {})
        chat_id = str(chat.get("id", ""))
        if not chat_id:
            continue
        title = (
            chat.get("title")
            or " ".join(
                item
                for item in [chat.get("first_name"), chat.get("last_name")]
                if item
            )
            or chat.get("username")
            or "unknown"
        )
        found[chat_id] = {
            "chat_id": chat_id,
            "type": str(chat.get("type", "")),
            "name": title,
        }

    return list(found.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    send_parser = sub.add_parser("send")
    send_parser.add_argument("message")

    sub.add_parser("get-chat-id")
    args = parser.parse_args()

    if args.command == "send":
        send(args.message, strict=True)
        print("Telegram message sent.")
    else:
        ids = get_chat_ids()
        if not ids:
            print(
                "No chats found. Send a message to your bot, then run this command again."
            )
            return
        print(json.dumps(ids, indent=2))


if __name__ == "__main__":
    main()
