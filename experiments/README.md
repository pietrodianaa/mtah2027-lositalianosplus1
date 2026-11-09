# experiments/

One folder per person (`experiments/<your-name>/`), never edited by anyone else. Notebooks and
scratch scripts live here; code that works moves into your own module in `possession/`.
Nothing in `possession/` may import from here.

## Your log: experiments/<your-name>/LOG.md
One row per idea tried, kept or not. One file per person, so logs never conflict. The tech-doc
owner compiles them at the end. Scores come ONLY from `python run.py --tune --role <your role>`
(external matches); always record the labels and reference versions it prints, and never compare
numbers across different versions.

| Date | Issue / PR | Change | Your role's score | labels_version | Kept? | Why / source |
|---|---|---|---|---|---|---|

Agents never run notebooks on the match data. Clear outputs before committing.
