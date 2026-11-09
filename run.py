"""Two jobs, kept apart on purpose (AGENTS.md section 0):

    python run.py --tune --role dead_ball     your OWN score on the external matches, with every
    python run.py --tune --role control       other role replaced by the truth (scores.py), so
    python run.py --tune --role postprocess   nobody else's merge can move it. Agents may run it.
    python run.py --tune --role full          the joint check, end to end (lead, at milestones).

    python run.py --final            labels for the provided game -> submissions/<TEAM_NAME>.csv,
                                     then the official validator. HUMANS ONLY: it reads the match.
    python run.py --final --refresh  same, re-parsing the raw tracking instead of cache/

There is deliberately no option to score against the public 5 minutes: we may not tune on it, and
tests/test_baseline_match.py already proves our frames are aligned with the official baseline.
Features have no --role: their owner checks each feature directly (see possession/features.py).
"""

import argparse
import os
import subprocess
import sys

from possession import config, external, io, pipeline

ROOT = os.path.dirname(os.path.abspath(__file__))


def tune(role):
    if not external.games():
        sys.exit("No external matches yet. Set up possession/external.py first.")
    result = external.evaluate(role)
    print(" ".join("{}={}".format(k, round(v, 4) if isinstance(v, float) else v)
                   for k, v in result.items()))


def final(refresh):
    labels = pipeline.run(refresh_cache=refresh)
    path = os.path.join(ROOT, "submissions", "{}.csv".format(config.TEAM_NAME))
    io.write_submission(path, labels)
    print("wrote", os.path.relpath(path, ROOT))
    subprocess.run([sys.executable, "evaluation/validate_submission.py", path],
                   cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tune", action="store_true", help="score one role on external matches")
    group.add_argument("--final", action="store_true", help="write the submission (humans only)")
    parser.add_argument("--role", choices=external.ROLES, help="role to score with --tune")
    parser.add_argument("--refresh", action="store_true", help="ignore cache/frames.npz")
    args = parser.parse_args()
    if args.tune:
        if not args.role:
            parser.error("--tune needs --role")
        tune(args.role)
    else:
        final(args.refresh)


if __name__ == "__main__":
    main()
