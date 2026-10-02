# Demo output

This folder is filled by `scripts/run_demo.sh`, which runs against the real Claude API. It needs
`ANTHROPIC_API_KEY` in `.env`. Nothing here is mocked; until the script runs, this folder holds only
this note.

After a run it contains:

- `plan.yaml`
- `episodes/ep_001.md` … `ep_015.md`
- `directives.md`: each intervention, the beats it re-planned, and the later episodes it influenced,
  with judge adherence scores
- `trace.jsonl`
- `stats.md`
- `estimate.md`
- `story.md`
