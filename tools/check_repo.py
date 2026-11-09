"""Repo guard: run by the pre-commit hook (staged files) and by GitHub Actions (whole repo).

    python tools/check_repo.py --staged     # what is about to be committed
    python tools/check_repo.py              # every tracked file (CI)

Fails if:
  1. a match-data file, cache, video, submission or external dataset would be committed;
  2. any tracked file is larger than 5 MB;
  3. an official upstream file differs from tools/official.sha256 (the scorer must stay
     identical to the organisers'). After pulling an organiser fix from upstream, the lead
     refreshes the list:  python tools/check_repo.py --refresh-official
Standard library only.
"""

import argparse
import fnmatch
import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL_LIST = os.path.join(ROOT, "tools", "official.sha256")
MAX_BYTES = 5 * 1024 * 1024

FORBIDDEN = [
    "data/*", "cache/*", "external_data/*", "submissions/*.csv", ".board/*",
    "evaluation/reports/*", "evaluation/scoreboard.csv",
    "*.mp4", "*.npz", "*.pkl", "*_atd.txt", "*_atd.xml", "*_smart_tagging.xml",
    "*_homography.csv",
]
ALLOWED = ["data/README.md", "*/.gitignore", "*/README.md"]

OFFICIAL = (["games.py", "README.md", "CHALLENGE.md", "DATA.md", "EVALUATION.md", "FAQ.md",
             "OVERVIEW.md", "RULES.md", "SUBMISSION.md", "LICENSE", "requirements.txt",
             "requirements-starter.txt"]
            + ["evaluation/" + f for f in ("possession_metrics.py", "run_evaluation.py",
                                           "validate_submission.py")]
            + ["evaluation/gt/BAR-ATH_possession_gt_public.csv",
               "evaluation/gt/BAR-ATH_possession_gt_public_meta.json"]
            + ["starter/" + f for f in ("baseline.py", "load_atd.py", "load_homography.py",
                                        "load_smart_tagging.py", "make_dummy_submission.py",
                                        "utils.py")])


def git_files(staged):
    cmd = (["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"] if staged
           else ["git", "ls-files"])
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line]


def sha(path):
    with open(os.path.join(ROOT, path), "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def refresh_official():
    with open(OFFICIAL_LIST, "w") as handle:
        for path in OFFICIAL:
            handle.write("{}  {}\n".format(sha(path), path))
    print("wrote", os.path.relpath(OFFICIAL_LIST, ROOT))


def check(staged):
    problems = []
    for path in git_files(staged):
        if any(fnmatch.fnmatch(path, p) for p in ALLOWED):
            continue
        if any(fnmatch.fnmatch(path, p) for p in FORBIDDEN):
            problems.append("data/derived file must not be committed: " + path)
        elif os.path.isfile(os.path.join(ROOT, path)) and \
                os.path.getsize(os.path.join(ROOT, path)) > MAX_BYTES:
            problems.append("file over 5 MB: " + path)

    if os.path.isfile(OFFICIAL_LIST):
        with open(OFFICIAL_LIST) as handle:
            for line in handle:
                digest, path = line.split()
                if not os.path.isfile(os.path.join(ROOT, path)):
                    problems.append("official file missing: " + path)
                elif sha(path) != digest:
                    problems.append("official file changed (never edit upstream files): " + path)
    else:
        problems.append("tools/official.sha256 missing")

    for problem in problems:
        print("BLOCKED:", problem)
    if problems:
        print("\nSee AGENTS.md section 0. Unstage with: git restore --staged <file>")
        return 1
    print("repo check OK")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--refresh-official", action="store_true")
    args = parser.parse_args()
    if args.refresh_official:
        refresh_official()
        sys.exit(0)
    sys.exit(check(args.staged))
