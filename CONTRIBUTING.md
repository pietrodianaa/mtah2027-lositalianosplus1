# How we work

Rules first: **AGENTS.md section 0** (organisers' clarification of 3 Nov 2026). In short: AI
agents never see the match data, nothing is tuned on the provided game or its ground truth, the
video is never used to check our output, and nothing about the match is made by hand.

## One-time setup (every member, about 20 minutes)
1. Registered on the hackathon form with the **exact team name** (closes Nov 10).
2. Install Python 3.11, git and the GitHub CLI (https://cli.github.com); run `gh auth login`.
3. Clone and install (yourself, not through an agent):
   ```
   git clone <our repo URL> mtah2027 && cd mtah2027
   python3.11 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
   pip install -r requirements-team.txt
   git config core.hooksPath .githooks                       # commit guard
   ```
4. **Yourself, by hand:** download from the Drive folder into `data/tactical/` (keep the names):
   `BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml`, `BAR-ATH_tactical_smart_tagging.xml`.
   Skip `broadcast/`. Never open them with an AI tool.
5. **Yourself:** `MTAH_HUMAN=1 python -m pytest -q` (Windows PowerShell: `$env:MTAH_HUMAN=1`
   first). The baseline test reads the match and checks our pipeline reproduces the official
   baseline frame for frame: all tests pass, none skipped. Agents run plain `pytest`, which
   skips it.
6. External data for `--tune`: run `tools/get_external_data.sh` once (installs git-lfs, fetches
   SkillCorner into gitignored `external_data/`, rebuilds the seeded labels; ends with `OK`).
7. `tools/board whoami`, then post `[Your name] set up, taking <role>` on the board (issue #1).

## Daily loop
```
git checkout main && git pull
# pick or open an issue (#12), say on the board you are taking it
git checkout -b 12-ball-out-of-bounds
# code in your own file; every new number at its top gets "# source: ..."
python -m pytest -q
python run.py --tune --role dead_ball # YOUR role's score on external matches (once set up)
git add possession/dead_ball.py experiments/<your-name>/LOG.md
git commit -m "Dead ball: ball out of bounds"          # commit guard runs here
git push -u origin HEAD               # open a PR with "Closes #12" + external scores
```
When the automatic checks are green and your role's gain is real (clearly above noise, or
clearly right football-wise), **merge it yourself** and post it on the board with the labels
version. Your score never depends on anyone else's work (AGENTS.md section 4).

## Who may run what
| Command | Agent | Human |
|---|---|---|
| `python -m pytest -q` (synthetic part) | yes | yes |
| `python run.py --tune --role <role>` (external matches) | yes | yes |
| `python run.py --final` (provided game) | **no** | yes, share only "valid / not valid" |
| Opening `data/`, `cache/`, notebooks on the match | **no** | yes, to understand the data |
| Watching the video | **no** | only to understand what counts as possession, never to check our labels |

## Never
- Tune or choose anything using the ground truth, the public 5 minutes, the video or the
  provided game. Every number is football reasoning or tuned on external data.
- Write down anything about the match by hand (labels, frame lists, "throw-in at frame X").
- Commit or post data, cache, videos, submissions or frame-level tables (the guard blocks most).
- Edit official files (`games.py`, `evaluation/`, `starter/`, official `*.md`). The checks fail.
