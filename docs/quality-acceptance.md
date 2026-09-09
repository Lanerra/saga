# Quality acceptance and advisory maintenance

The workflow records quality policy in chapter state, then retains the decision with the accepted attempt. A generation prompt is guidance, not proof that a quality check completed. A `completed` check means its implemented protocol ran, not that a model proved the manuscript correct.

## Policy

- `QUALITY_ACCEPTANCE_POLICY=author` (default) preserves author force-continue and maximum-revision acceptance of observed quality findings. These decisions are `accepted_with_exceptions`, not clean passes. Failed prose evaluation requires explicit force-continue; reaching a revision limit alone does not certify an incomplete evaluation.
- `QUALITY_ACCEPTANCE_POLICY=strict` requires completed validation and passing quality gates. Force-continue and maximum-revision controls cannot override failures.
- `QA_ACCEPTANCE_POLICY=advisory` (default) allows graph-quality findings, query failures, disabled checks and cadence skips with explicit exceptions. `mandatory` requires the read-only graph-quality check to complete without findings before manuscript preparation; cadence cannot defer it. Disabling a mandatory check blocks acceptance.
- Graph healing, relationship deduplication and consolidation remain postpublication advisory maintenance under both policies. Their errors do not retroactively invalidate manuscript acceptance; their separate receipts preserve what actually happened.
- Missing/stale checks, malformed candidate or prior-canon input, fatal errors, incomplete extraction and compensation barriers are never author quality exceptions. No gate changes B16's prior-canon authority or canonical-alias interpretation.

The prose threshold preserves the existing mean of coherence, prose-quality and plot-advancement scores. All five scores (including pacing and tone), feedback and findings are retained. Malformed responses no longer invent passing fallback scores. Failed evaluations record null scores and an explicit failure reason. Evaluation evidence records evaluated and draft character counts: the current prompt samples the first and last 4000 characters when the draft exceeds 8000 characters. This is not whole-chapter coverage.

## Receipts and recovery

Lifecycle acceptance stores `quality` in `.saga/attempts/<attempt-id>/acceptance.json` and in the same graph acceptance payload. It contains versioned policy, source checksums, check completion/reasons/coverage, scores, feedback, findings, force/revision controls, exceptions and status. Receipt replay checks the retained policy and draft/extraction identity, rather than today's settings. Conflicting checkpoint quality evidence fails closed. Graph/file acknowledgement and chapter advancement remain the existing lifecycle protocol.

The legacy direct finalizer also gates publication. It stores a `.quality.json` sidecar beside the retained `.md` manuscript under `chapters/.manuscripts/chapter_NNN/`. The sidecar binds the exact manuscript receipt. A different decision cannot overwrite the same sidecar; a distinct explicitly evaluated manuscript version has its own receipt. A prepared quality decision is not itself proof of graph acknowledgement: use the existing accepted-manuscript receipt as the publication selector.

Postpublication maintenance stores content-addressed JSON observations under the lifecycle attempt's `maintenance/`, or `<manuscript-sha>.maintenance/` for policy-bearing legacy publication. Errors, warnings and findings are labeled `accepted_with_exceptions`. Standalone legacy diagnostics with no publication policy remain diagnostics only and cannot confer acceptance.

Old lifecycle acceptance receipts without quality evidence cannot be replayed as newly validated acceptance. Old checkpoints cannot finalize without running explicit validation. Existing manuscript readers retain their historical storage contract; they do not retroactively attest that an old chapter passed these new gates. No automatic migration or revalidation command is introduced here. Preserve old artifacts and arrange explicit operator reconciliation; do not edit receipt JSON to manufacture a pass.

The receipts are human-readable evidence. Both finalization routes emit an `accepted_with_exceptions` warning with policy, reasons and receipt path after verified publication. The existing rich CLI has not been redesigned to summarize them. Downstream UI work should read the retained status and reasons, not infer a clean pass from cleared `needs_revision` state.
