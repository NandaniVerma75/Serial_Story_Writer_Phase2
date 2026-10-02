# Design decisions

**How does the system remember the story at episode 150?** It never puts the full story in context.
The context for episode N is rebuilt from layered memory under a ~12k-token budget, in priority order:

1. The bible, held in a cached system prompt.
2. Every active human directive.
3. Beat N, beats N+1..N+3, the arc goal, and any scheduled fates.
4. The full text of episode N-1.
5. Summaries of episodes N-5..N-2.
6. The "story so far" summary, refreshed at arc ends, plus rolling arc summaries updated every 5 episodes.
7. Retrieved character cards: characters named in the beat, active in the last 3 episodes, or related to those.
8. Threads that are due or overdue, plus a "don't forget" list of threads untouched for more than 15 episodes.
9. Ledger facts about the retrieved characters and places.

Layers 1–3 are never dropped. The rest are trimmed from the bottom, and every drop is logged.

All derived state is **event-sourced**: each fact, status change, relationship change, thread event
and timeline entry carries the `episode_id` that produced it. Memory is queried *as of* episode N.
That makes retroactive edits mechanical: delete everything at or after K, re-extract K, replay the
stored extractions for K+1.. (no LLM cost), then consistency-check the later episodes against the
rebuilt canon and let the human choose keep, auto-revise, or regenerate.

Facts carry supersession: a new value for the same (subject, predicate) closes the old one. Thread
references are stored by description, not id, so replay survives the ids changing.

**Where does the human step in, and why there?**

- **Plan approval.** Fixing a plan is cheapest before any prose exists.
- **Every episode before commit.** Only approved text becomes canon and gets extracted, so a bad
  draft cannot pollute memory.
- **Feedback at any time.** Feedback becomes a scoped, persistent directive. If it needs a re-plan,
  the human approves the beat diff.
- **Retroactive edits.** The human decides the downstream policy.

The machine does the bounded work first (checks plus at most 2 revisions, with a per-episode cost
cap), so the human sees only drafts that already pass, or a draft with its specific failures listed.
Propagation is measured, not assumed: each episode records which directives were in its context and
the judge's per-directive adherence score (`story directives --impact`).

**How do you detect inconsistency or repetition before a human has to?**

- **Code checks** (cheap, deterministic):
  - word count;
  - dead characters listed in the scene plan, or written with action/speech verbs;
  - unknown or near-misspelled names;
  - a timeline that runs backwards without a flashback flag;
  - banned tics;
  - overdue fates.
- **LLM checks** (cheap model, given the same retrieved canon the writer saw):
  - **Continuity editor**: returns contradictions with evidence.
  - **Rubric judge**: scores hook, momentum, voice, beat adherence and directive adherence. Its
    summary of the draft is embedded and compared by cosine similarity against every past episode
    summary.
- **Hook variety**: the hook type must differ from the previous episode's, and repeating one from the
  last 3 raises a warning.

Plan-level failures are repaired in code (clamp the day, drop a dead character) before the reviser
is asked to fix the prose.

**What breaks first as the story grows, and how would I fix it?**

1. **Compounding extraction errors and summary drift.** A wrong fact extracted at episode 20 becomes
   "canon" for 180 episodes, and summaries of summaries lose nuance. Fixes:
   - store confidence and spot-verify low-confidence facts against the source text;
   - periodically rebuild "story so far" from episode summaries rather than from the previous summary;
   - let the human browse and correct facts (`story show facts`) with the same reconcile path.
2. **Fact-ledger bloat and retrieval precision.** Keyword and name retrieval works for tens of
   characters, but by ep 150 there are thousands of facts. Fixes: embed facts and retrieve by
   similarity to the scene plan, cluster facts by entity, and expire trivial facts.
3. **Plan rigidity vs. an emergent story.** Beats are written before the story exists. Directive
   re-plans patch a 20-episode window, but nothing re-plans an arc when the story drifts. Fix: an
   arc-boundary re-plan step that compares the arc summary with its planned goal.
4. **Cost of re-checking after retro edits.** The cost grows linearly with downstream episodes, so it
   is capped at a 20-episode window (`--window`); replay itself is free. The fix would be to re-check
   only episodes whose retrieved entities intersect the edited facts.
5. **LLM-judge leniency.** Same-family judges are generous, and the rubric is calibrated by prompt
   alone. Fixes:
   - track the judge's pass rate against human reject rate in the traces;
   - use a different model family for judging;
   - add pairwise comparison against the previous episode.
6. **The dead-character regex** can flag remembered actions ("the way Meera laughed"). It is a cheap
   pre-filter; the continuity check and the human make the final call.

**Other choices.**

- **No agent framework.** Each step is a plain function, and the stages persist to SQLite (`planned`,
  `drafted`, `review`, `approved`). That gives exact resume and makes the steps testable.
- **Hierarchical planning** takes 13 calls, and code (not the model) enforces the 200-beat numbering.
- **Token counts for budgeting** are estimated at about 4 characters per token. Billed tokens come
  from provider usage.
- **Embeddings.** Anthropic has no embeddings endpoint, so the system uses OpenAI's if a key is set,
  and otherwise a local hashed n-gram embedding. That is weaker on paraphrase but good at catching
  repeated events.
- **Model tiers.** Opus 5.5 handles planning and prose; Haiku 4.5 handles all checking, extraction and
  summaries.
- **Caching and fallbacks.** The bible-bearing system prompt is marked for prompt caching.
  Server-side refusal fallbacks are on for Opus 5.5, and switch off automatically if the account
  doesn't support them.
