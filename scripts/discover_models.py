#!/usr/bin/env python3
"""Ask each provider what it actually serves, and print registry entries for it.

    docker compose exec -T ai-agent python scripts/discover_models.py
    docker compose exec -T ai-agent python scripts/discover_models.py --merge

Why this exists
---------------
A registry hand-written from memory goes stale the week after it is written, and
a wrong model id fails at the API with a confusing error. The providers publish
the authoritative list; this reads it with the keys already in .env and emits
JSON in exactly the shape `llm_registry.json` wants.

`--merge` writes the new identifiers straight into the registry, keeping every
entry that is already there. Existing mappings are never overwritten: a
deliberate choice you made beats anything discovered here.

Reads keys from the environment. Never prints them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.services import models  # noqa: E402

#: Provider -> (models endpoint, key). OpenAI-compatible listing on both.
ENDPOINTS = {
    "groq": ("https://api.groq.com/openai/v1/models", lambda: settings.groq_api_key),
    "openai": ("https://api.openai.com/v1/models", lambda: settings.openai_api_key),
}


def identifier_for(provider: str, model_id: str) -> str:
    """`openai/gpt-oss-120b` on groq -> `groq_gpt_oss_120b`.

    Mirrors the naming a dashboard tends to use: provider, then the model with
    punctuation flattened to underscores. Vendor prefixes inside the id
    (`openai/`, `meta-llama/`) are dropped - they describe who made the model,
    not who serves it, and the provider is already the first word.
    """
    tail = model_id.split("/")[-1]
    slug = re.sub(r"[^a-z0-9]+", "_", tail.lower()).strip("_")
    return f"{provider}_{slug}"


def fetch(provider: str) -> list[str]:
    url, get_key = ENDPOINTS[provider]
    key = get_key()
    if not key:
        print(f"  {provider}: no API key configured - skipping", file=sys.stderr)
        return []

    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        # Deliberately does not echo the body: it can contain the key.
        print(f"  {provider}: HTTP {error.code} listing models - skipping", file=sys.stderr)
        return []
    except Exception as error:  # noqa: BLE001
        print(f"  {provider}: {type(error).__name__} listing models - skipping", file=sys.stderr)
        return []

    ids = [str(item.get("id")) for item in payload.get("data", []) if item.get("id")]
    return sorted(ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merge",
        action="store_true",
        help="write new identifiers into the registry (never overwrites existing ones)",
    )
    args = parser.parse_args()

    registry = models.registry()
    existing = dict(registry.models)

    discovered: dict[str, dict[str, str]] = {}
    print("Querying providers...", file=sys.stderr)
    for provider in ENDPOINTS:
        if provider not in registry.providers:
            continue
        for model_id in fetch(provider):
            discovered[identifier_for(provider, model_id)] = {
                "provider": provider,
                "model": model_id,
            }

    new = {k: v for k, v in discovered.items() if k not in existing}

    print(f"\n{len(discovered)} models live, {len(new)} not yet registered.\n", file=sys.stderr)
    print(json.dumps(new, indent=2, sort_keys=True))

    if not args.merge:
        print(
            "\nPaste these into the 'models' object of "
            f"{registry.source}, or re-run with --merge.",
            file=sys.stderr,
        )
        return 0

    path = Path(registry.source)
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Discovered entries go UNDER the hand-written ones: an id you chose
    # deliberately must win over one derived from a listing.
    raw["models"] = {**new, **raw.get("models", {})}
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    print(f"\nMerged {len(new)} entries into {path}. Rebuild the image.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
