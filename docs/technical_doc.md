# <TEAM_NAME> — Technical document, More than a Hack 2027 Phase 1

*A few pages. Owner: external data + doc. Every owner adds notes to their section as they go.*

## 1. Approach
What we built and why. One diagram: features → dead ball + control → post-processing.

## 2. Assets used
Provided assets used as input and how; which we tried and dropped.

## 3. Compliance with the rules clarification (3 Nov 2026)
- No training, fitting or tuning on the ground truth, the public 5 minutes, the video or the
  provided game. Table: every parameter in `config.py` with its source.
- External data used for tuning: name, link, licence or permission, how possession labels were
  derived from its events (by code).
- No hand-made data about the match. No checking of our output against the video.
- AI tools: what we used them for, and that they never received the provided data. Disclose:
  on Oct 31, before the rules clarification, an AI assistant briefly read the tracking metadata
  (BAR-ATH_tactical_atd.xml) and the Smart Tagging file to explain their format; the copies were
  deleted and nothing was used for training or tuning.

## 4. Possession rules implemented
- Dead ball · control / change of possession (loose balls, deflections, aerial duels) ·
  restarts · smoothing and abstention (`X`) policy.

## 5. Results on external data
Each role's own score (other roles replaced by truth), then the joint end-to-end score.
Labels version used. What each component added.

## 6. Limitations
Where it fails and what we would fix with more time.

## 7. Reproduction
```
pip install -r requirements-team.txt
# put the data bundle in data/ (see DATA.md)
python run.py --final        # -> submissions/<TEAM_NAME>.csv
```
