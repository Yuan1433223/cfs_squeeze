"""Probe ModelScope for an MMBench-flavoured dataset that actually exists.

Run this once on the cloud DSW; paste the output back. The script tells us:
    1. which candidate repo IDs return HTTP 200 on the ``/repo`` endpoint
       (i.e. the dataset exists and is reachable);
    2. for each existing repo, the first ~80 file paths under it (so we can
       pick the real dev parquet/tsv path).

We then hard-code the winner into ``_load_mmbench``.
"""
from __future__ import annotations

import json

import requests


CANDIDATES = [
    "lmms-lab/MMBench",                  # confirmed accessible at modelscope.cn/datasets/lmms-lab/MMBench
    "AI-ModelScope/MMBench",
    "AI-ModelScope/MMBench_DEV_EN",
    "AI-ModelScope/MMBench-V11",
    "modelscope/MMBench",
    "OpenCompass/MMBench",
    "OpenGVLab/MMBench",
    "Shanghai_AI_Laboratory/MMBench",
]


def info(repo: str):
    """Hit /api/v1/datasets/<repo> -- a 200 means the repo id resolves."""
    try:
        r = requests.get(f"https://www.modelscope.cn/api/v1/datasets/{repo}", timeout=20)
        return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else None)
    except Exception as e:
        return f"ERR:{type(e).__name__}", None


def list_files(repo: str, page_size=100):
    """Try the file-listing variants we've seen in the wild."""
    urls = [
        f"https://www.modelscope.cn/api/v1/datasets/{repo}/repo/tree?Recursive=true&PageSize={page_size}",
        f"https://www.modelscope.cn/api/v1/datasets/{repo}/repo/files?Revision=master&Recursive=true",
        f"https://www.modelscope.cn/api/v1/datasets/{repo}/repo/tree?Revision=master&Recursive=true",
    ]
    for u in urls:
        try:
            r = requests.get(u, timeout=30)
        except Exception as e:
            continue
        if r.status_code != 200:
            continue
        try:
            j = r.json()
        except Exception:
            continue
        # Walk a few likely shapes.
        for path in (("Data", "Files"), ("Data", "files"), ("data", "files"), ("files",), ("Data", "tree")):
            cur = j
            ok = True
            for k in path:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    ok = False
                    break
            if ok and isinstance(cur, list):
                return [
                    f.get("Path") or f.get("path") or f.get("name")
                    for f in cur if isinstance(f, dict)
                ]
        return ["<unparseable>", json.dumps(j)[:300]]
    return None


def main():
    print("=" * 70)
    print("MMBench candidate scan")
    print("=" * 70)
    found = []
    for repo in CANDIDATES:
        status, _ = info(repo)
        marker = "OK " if status == 200 else "    "
        print(f"  {marker}{status:<4} {repo}")
        if status == 200:
            found.append(repo)
    if not found:
        print("\nNo candidate repo resolved. The MMBench id on this DSW is unknown -- "
              "open https://www.modelscope.cn/datasets and search for 'MMBench' manually, "
              "then add the result to _load_mmbench candidates.")
        return
    print()
    for repo in found:
        files = list_files(repo)
        print(f"--- files in {repo} (first 80) ---")
        if not files:
            print("  <listing endpoint blocked or unsupported>")
            continue
        for fp in files[:80]:
            print(f"  {fp}")


if __name__ == "__main__":
    main()
