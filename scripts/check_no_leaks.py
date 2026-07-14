#!/usr/bin/env python3
"""Pre-commit / pre-push guard against corporate-data leaks.

The qwen3_vl repository is public. This scanner refuses to let private
information reach a commit:

  * internal storage paths (/mnt/r3, /mnt/hdd1, /mnt/nvme*, ros_logs, ...)
  * corporate dataset / project identifiers (Aurus, NAMI, avd share, ...)
  * raw media captured from internal logs (jpg/png/mp4/db3/rosbag/pcd/bin)
  * hostnames, IPs and credentials that only belong on the GPU servers

Run it manually or wire it into a pre-push hook:

    python3 scripts/check_no_leaks.py [--staged | --worktree | PATH...]

Exit code 0 = clean, 1 = leaked content detected.

Allowed binary fixtures (small public evaluation samples committed on purpose)
live under evaluation/ and tests/ and are whitelisted by path.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# Pathname / identifier fragments that only ever appear in internal context.
FORBIDDEN_TOKENS = (
    "Aurus",
    "/mnt/r3",
    "/mnt/hdd1",
    "/mnt/hdd2",
    "/mnt/nvme",
    "/mnt/data2",
    "ros_logs",
    "rosbag",
    "NAMI",
    "192.168.11.227",
    "10.152.1.180",
    "ru.tuna.am",
    "graphicsserver",
    "PMRepnikov",
    "evinogradov",
    "silin",
    "//192.168",
    "avd",
    "erTgfD",
    "juik122",
)

# Credential-shaped secrets: SSH passwords, CIFS passwords, GitHub PATs.
SECRET_PATTERNS = (
    re.compile(r"github_pat_[0-9A-Za-z_]{20,}"),
    re.compile(r"\b(erTgfD\w*|juik122)\b"),
    re.compile(r"//192\.168\.\d+\.\d+/[A-Za-z0-9_.$-]+"),
)

# Binary / media extensions that must never be committed from internal captures.
MEDIA_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif",
    ".mp4", ".mov", ".mkv", ".avi", ".webm",
    ".db3", ".mcap", ".bag", ".rosbag", ".pcd", ".bin", ".npy",
}

# Paths that are allowed to ship small committed fixtures (kept minimal),
# or that legitimately mention the forbidden tokens (this scanner, gitignore).
WHITELIST_GLOBS = (
    "evaluation/**",
    "tests/**",
    "docs/**/*.png",
    "demo/web/**",
    "scripts/check_no_leaks.py",
    "scripts/extract_frames.py",
    ".gitignore",
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _is_whitelisted(path: Path) -> bool:
    try:
        rel = path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return False
    from fnmatch import fnmatch
    return any(fnmatch(rel, pattern) for pattern in WHITELIST_GLOBS)


def _text_leaks(text: str, source: str) -> list[str]:
    findings: list[str] = []
    lowered = text
    for token in FORBIDDEN_TOKENS:
        if token.lower() in lowered.lower():
            findings.append(f"forbidden token {token!r}")
    for pattern in SECRET_PATTERNS:
        match = pattern.search(text)
        if match:
            findings.append(f"secret-like pattern matching {match.group(0)!r}")
    return findings


def _scan_file(path: Path) -> list[str]:
    findings: list[str] = []
    if not path.is_file():
        return findings
    # Whitelisted files (fixtures, this scanner, gitignore) are trusted as-is.
    if _is_whitelisted(path):
        return findings
    if path.suffix.lower() in MEDIA_EXTENSIONS:
        findings.append(f"media/binary extension {path.suffix} (not whitelisted)")
        return findings
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return findings
    for finding in _text_leaks(text, str(path)):
        findings.append(finding)
    return findings


def _collect_targets(mode: str, explicit: list[str]) -> list[Path]:
    if explicit:
        return [Path(item).resolve() for item in explicit]
    if mode == "staged":
        import subprocess
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            cwd=REPO_ROOT, text=True, capture_output=True, check=False,
        )
        return [REPO_ROOT / line for line in result.stdout.splitlines() if line]
    # worktree (default): scan tracked + untracked, excluding ignored files.
    import subprocess
    result = subprocess.run(
        ["git", "status", "--porcelain", "-uall", "--untracked-files=all"],
        cwd=REPO_ROOT, text=True, capture_output=True, check=False,
    )
    paths: list[Path] = []
    for line in result.stdout.splitlines():
        entry = line[3:].strip().strip('"')
        if entry:
            paths.append(REPO_ROOT / entry)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--staged", action="store_true", help="scan git-index-staged files only")
    source.add_argument("--worktree", action="store_true", help="scan the whole working tree (default)")
    parser.add_argument("paths", nargs="*", help="explicit paths to scan")
    args = parser.parse_args(argv)

    mode = "staged" if args.staged else "worktree"
    targets = _collect_targets(mode, args.paths)

    leaks: list[tuple[str, str]] = []
    for target in targets:
        for finding in _scan_file(target):
            try:
                rel = target.relative_to(REPO_ROOT)
            except ValueError:
                rel = target
            leaks.append((str(rel), finding))

    if not leaks:
        print("check_no_leaks: clean (no forbidden tokens, secrets, or media)")
        return 0

    print("check_no_leaks: LEAK DETECTED — refusing to proceed:", file=sys.stderr)
    for rel, finding in leaks:
        print(f"  {rel}: {finding}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
