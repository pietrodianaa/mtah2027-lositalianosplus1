"""Team possession pipeline for More than a Hack 2027, Phase 1.

    io -> features -> dead_ball + control -> postprocess -> CSV

Each stage is one module with one public function and a fixed input/output contract (see
CLAUDE.md), so each workstream can change its own module without touching the others.
"""
