"""General facts about the data, owned by the lead. NOT for role parameters.

Each role keeps its own numbers at the top of its own module (dead_ball.py, control.py,
features.py, postprocess.py), so no two people ever edit the same file. Every number, here or
there, carries a `# source:` comment, one of:
  - "football reasoning: <why>"            chosen from first principles, not fitted
  - "external: <dataset> v<labels version>" tuned with `python run.py --tune --role <role>`
  - "official baseline default"            copied from the organisers' starter code
  - "data format" / "DATA.md"              a fact about the files, not a choice
Rule (organisers, 3 Nov 2026): nothing is tuned on the ground truth, the public 5 minutes, the
video or the provided game. A number without a source does not get merged (a test checks it).
"""

FPS = 25                       # source: data format
N_FRAMES = 199319              # source: data format (tactical video length = submission rows)
PITCH = (105.3, 68.0)          # source: data format (games.py, metres)
TEAM_NAME = "lositalianos+1"        # <- set once; used for submissions/<TEAM_NAME>.csv

# Speeds in the raw ATD are km/h (atd.xml: "speed_kph"), NOT m/s as DATA.md says.
KPH_TO_MS = 1 / 3.6            # source: data format
