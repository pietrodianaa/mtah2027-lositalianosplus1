# Team rules for every agent (and human)

Read with CLAUDE.md (task context). Workflow adapted from
https://github.com/azizu06/hackathon-agent-board.

## 0. Competition rules (override everything below)

From the organisers' clarification of 3 Nov 2026, which overrides all earlier documentation.

- **Agents never read the match data.** Nothing under `data/`, `cache/`, `evaluation/gt/`,
  `evaluation/results/`, `evaluation/reports/` or `submissions/`, no videos, and no command
  that prints per-frame values from them (`head`, `cat`, `print(df)`, notebook tables).
  Agents write code and run `python -m pytest -q` (synthetic data). **A human runs anything on
  the real match** and shares only aggregate results. (`.claude/settings.json` also blocks reads.)
- **No tuning on the provided game.** Every parameter is chosen from football reasoning or tuned
  on external data (`possession/external.py`). Never on the ground truth, the public 5 minutes,
  the video, or the provided game's data.
- **No checking our output against the video.** The video may be watched only to understand
  what counts as possession, never to look at our labels.
- **No hand-made data about the match**: no labels, frame lists or notes like "throw-in at
  frame X". Labels come from code.
- **Every number states its source** (`# source: ...`), in your own module.
- Never commit or post data, frame-level tables, clips or videos. The repo and the board become
  public at submission.

## 1. The board (pinned GitHub issue "Message Board", #1 unless BOARD_ISSUE says otherwise)

Agents read and post via `tools/board` (needs only `gh`, logged in). Humans can post directly.

- Start every post with `[<human>'s <agent>]`, e.g. `[Pietro's Claude]`. Humans: `[Pietro]`.
- Keep it short: what changed, which file or contract, what you need, from whom.
- Mention by GitHub login: `@pietrodianaa` (Pietro, lead: integration + post-processing),
  `@<login-A>` (Dead ball), `@<login-B>` (Control), `@<login-C>` (Features),
  `@<login-D>` (External data + doc), `@all` (everyone).
- **Mentions you or `@all`:** act on it and reply.
- **Not addressed to you:** read it; change course only if it touches a file or contract you own.
  No acknowledgement-only replies.
- When you merge, post: what changed and **your role's own score** from
  `python run.py --tune --role <role>` with its labels version (never the public 5 minutes).

Commands: `tools/board unread` (new posts, flags MENTION) · `tools/board read -n 20` ·
`tools/board post "[Name's Agent] text"` · `tools/board wait` (blocks until someone else
posts) · `tools/board whoami`.

Staying in the loop: Claude Code's hooks (`.claude/settings.json`) show unread posts at session
start and on every prompt, and ask the agent to keep one background `tools/board wait`
running. Waiting costs nothing; the agent only works when a post arrives. Set `BOARD_WAIT=off`
in your shell to opt out. Other agents: run `tools/board unread` at task start, before each
commit, and after merges.

## 2. Chain of command

Humans over agents. Pietro is team lead: only he reassigns work between teammates. Changes to
integration files need his approval (CODEOWNERS enforces it).

## 3. Workflow

1. Pick or open a GitHub issue for the task; say on the board that you are taking it.
2. Branch `<issue#>-<slug>` from fresh `main`.
3. Edit only the file(s) you own (table below). Your numbers live at the top of your own module,
   each with its `# source:`. Need a change in someone else's file? Ask the owner on the board.
4. Stage only your own files (no `git add -A`). Run `python -m pytest -q`.
5. Open a PR with `Closes #N`. Its description: what changed, your role's score before and after
   (`python run.py --tune --role <role>`, same labels version), and the source of every number.
6. **The author merges their own PR** once the automatic checks are green and, for a score
   change, the gain is above the noise (about 0.04 spellF1) or clearly right football-wise.
   Then post on the board and add a row to your own `experiments/<your-name>/LOG.md`.
7. Always `git fetch` and rebase before pushing. Never force-push shared branches.
8. If a merge changes a contract others depend on (a feature name, a function signature),
   post on the board when it lands.

## 4. Independence: nobody's work moves anyone else's

- **Own file, own numbers.** Each role's parameters live in its own module; `config.py` holds only
  data facts (lead). No shared files except by agreement.
- **Own score.** `python run.py --tune --role <role>` scores your role with every OTHER role
  replaced by the truth or a frozen reference (`possession/scores.py`). Another person's merge
  cannot change your number. A test enforces this.
- **Features are add-only.** Once merged, a feature is never renamed, removed or changed in
  meaning; improvements get a new name (`ball_v2`). `tests/feature_contract.txt` lists them.
  Consumers switch to a new feature only when they choose to.
- **External labels are versioned.** `external.LABELS_VERSION` changes only with a post on the
  board; scores from different versions are never compared. Same for `scores.REFERENCE_VERSION`
  (changing `scores.py` needs the whole team's agreement).
- **Joint check at milestones only.** The lead runs `--role full` on Nov 12 (v1) and during
  Nov 13–14 to see how the parts combine. Nobody tunes against the full score in between.

## 5. Who owns what

| Area | Owner (GitHub login) | Files |
|---|---|---|
| Lead: integration + post-processing | `@pietrodianaa` | `possession/io.py`, `config.py`, `pipeline.py`, `postprocess.py`, `scores.py` (team-agreed), `run.py`, `tests/`, `.github/`, `tools/`, `.claude/` |
| Dead ball | `@<login-A>` | `possession/dead_ball.py` |
| Control / possession changes | `@<login-B>` | `possession/control.py` |
| Features | `@<login-C>` | `possession/features.py`, `tests/feature_contract.txt` (add-only) |
| External data + tech doc | `@<login-D>` | `possession/external.py` (loading and labels), `docs/technical_doc.md` |
| Everyone | — | only their own `experiments/<your-name>/` (notebooks + `LOG.md`) |

Never edit: `games.py`, `evaluation/`, `starter/`, the official `*.md` docs (upstream files; the
automatic checks fail if they change).
