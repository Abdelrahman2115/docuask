"""Deploy DocuAsk to a Hugging Face Docker Space.

    python scripts/deploy_space.py              # creates/updates <your-username>/docuask
    python scripts/deploy_space.py --name my-space-name

Needs in .env:
    HF_TOKEN=hf_...      (a *write* token from https://huggingface.co/settings/tokens)
    GROQ_API_KEY=...     (copied into the Space as an encrypted secret, never into files)

Why the API instead of `git push`: the Hub rejects plain-git pushes of binary
files (our PDFs) unless they go through its large-file storage; upload_folder
handles that automatically.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from docuask.config import PROJECT_ROOT, get_settings

GITHUB_URL = "https://github.com/Abdelrahman2115/docuask"

UPLOAD = ["Dockerfile", ".dockerignore", "requirements-docker.txt", "config.yaml", "pyproject.toml",
          "src/*", "scripts/*", "app/*", "data/sources.csv", "data/raw/*"]
SKIP = ["*__pycache__*", "*.pyc", "*.egg-info*", "*.bat", "*_log.txt"]

SPACE_README = f"""---
title: DocuAsk
emoji: 📞
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Cited answers about U.S. telecom consumer rights (FCC guides)
---

# DocuAsk

Retrieval-augmented assistant that answers questions about U.S. telecom consumer
rights from 40 FCC consumer guides, cites the exact page or section, and says
"I don't know" when the guides don't cover the question.

Stack: multilingual-e5-small embeddings · Chroma · Groq `gpt-oss-20b` · Gradio.

Code, evaluation and results: {GITHUB_URL}
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default=os.getenv("HF_SPACE_NAME", "docuask"))
    p.add_argument("--private", action="store_true")
    p.add_argument("--no-wait", action="store_true", help="don't wait for the build to finish")
    args = p.parse_args()

    s = get_settings()                       # loads .env
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN missing in .env (create a WRITE token at https://huggingface.co/settings/tokens)")
        return 1
    if not s.llm.groq_api_key:
        print("GROQ_API_KEY missing in .env")
        return 1

    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    user = api.whoami()["name"]
    repo_id = f"{user}/{args.name}"
    print(f"Space: https://huggingface.co/spaces/{repo_id}")

    api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    api.add_space_secret(repo_id, "GROQ_API_KEY", s.llm.groq_api_key,
                         description="Groq API key used by the app at runtime")
    print("Secret GROQ_API_KEY set (stored encrypted by Hugging Face, not in the repo).")

    api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(PROJECT_ROOT),
                      allow_patterns=UPLOAD, ignore_patterns=SKIP,
                      commit_message="Deploy DocuAsk")
    api.upload_file(repo_id=repo_id, repo_type="space", path_or_fileobj=SPACE_README.encode("utf-8"),
                    path_in_repo="README.md", commit_message="Space card")
    print("Files uploaded. The Space is now building the Docker image (usually 5-15 minutes).")

    if args.no_wait:
        return 0
    last, t0 = None, time.time()
    while time.time() - t0 < 30 * 60:
        runtime = api.get_space_runtime(repo_id)
        stage = str(getattr(runtime.stage, "value", runtime.stage))
        if stage != last:
            print(f"  [{(time.time() - t0) / 60:4.1f} min] {stage}", flush=True)
            last = stage
        if stage == "RUNNING":
            print(f"\nLive: https://huggingface.co/spaces/{repo_id}")
            return 0
        if stage in ("BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"):
            print(f"\nFailed with {stage}. Open the Space page -> 'Logs' to see why, and paste them here.")
            return 1
        time.sleep(20)
    print("Still building after 30 minutes; check the Space page.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
