#!/usr/bin/env bash
# End-to-end demo: full 200-episode plan, 15 episodes, two HITL interventions, a quit + resume.
# Usage: scripts/run_demo.sh ["<premise>"]
set -euo pipefail
cd "$(dirname "$0")/.."
STORY=${STORY:-.venv/bin/story}
ID=${DEMO_ID:-demo-story}
PREMISE=${1:-"Mumbai, 1994. Ravi, a 22-year-old dabbawala who delivers lunches across the city by bicycle and local train, finds a ledger hidden in a tiffin meant for a dead man. The ledger records bribes paid by a powerful builder. Meera, a sharp court stenographer who has loved Ravi quietly since school, realises the ledger can stop the demolition of their chawl - if they survive the people looking for it."}

rm -rf "stories/$ID"
echo "== 1. Plan: bible -> 5 acts -> 10 arcs -> 200 beats (auto-approved for the demo)"
$STORY new "$PREMISE" --id "$ID" --yes

echo "== 2. Episodes 1-4"
$STORY write --auto 4 --on-fail approve -s "$ID"

echo "== 3. Intervention 1 (after ep 4): pacing directive"
$STORY feedback "Slow down the romance between Ravi and Meera: no confession, kiss or declared feelings before episode 40. Keep it to small gestures, misread signals and things left unsaid." --yes -s "$ID"

echo "== 4. Episodes 5-8"
$STORY write --auto 4 --on-fail approve -s "$ID"

VICTIM=$(.venv/bin/python - "$ID" <<'PY'
import sys
from serial_writer.story import Story
s = Story.open(sys.argv[1])
names = [c.name for c in s.bible.characters]
skip = {n for n in names if n.split()[0].lower() in ("ravi", "meera")}
print(next((n for n in names[1:] if n not in skip), names[-1]))
PY
)
echo "== 5. Intervention 2 (after ep 8): character fate -> kill off $VICTIM by episode 11"
$STORY feedback "Kill off $VICTIM by episode 11. The death must happen on-page, be a consequence of the ledger, and change how the others act afterwards." --yes -s "$ID"

echo "== 6. Episodes 9-12, then the process exits (simulated quit)"
$STORY write --auto 4 --on-fail approve -s "$ID"

echo "== 7. Resume in a fresh process and continue to episode 15"
$STORY resume "$ID" --auto 3 --on-fail approve

echo "== 8. Collect demo artefacts into demo/"
rm -rf demo && mkdir -p demo
cp "stories/$ID/plan.yaml" demo/plan.yaml
cp -r "stories/$ID/episodes" demo/episodes
cp "stories/$ID/trace.jsonl" demo/trace.jsonl
$STORY directives --impact --md demo/directives.md -s "$ID"
$STORY stats --md demo/stats.md -s "$ID"
$STORY estimate --md demo/estimate.md -s "$ID"
$STORY export --out demo/story.md -s "$ID"
echo "Done. See demo/"
