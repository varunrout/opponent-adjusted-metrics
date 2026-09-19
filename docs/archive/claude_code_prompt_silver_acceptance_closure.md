# Task: Close the Silver v1 Data-Engineering Acceptance Gap (blocking, do this before any further feature/model work)

## Context

This project's Data Engineering acceptance has been sitting CONDITIONAL since ~19-20 August. The originally-published canonical Silver prefix had a verified `_SUCCESS` marker-ordering defect (the completion marker was not the last object uploaded, so a reader could theoretically observe a partial publish as "complete"). The stated remediation was: (1) patch the uploader to write `_SUCCESS` strictly last, (2) publish a fresh compliant Silver output preserving the old noncompliant prefix as immutable evidence (do not delete it), (3) verify `oam_core` reconciles against the new canonical output, (4) run the FULL repository test suite (not a focused subset) and confirm it passes. Only after all four gates pass does Data Engineering formally close and every downstream phase (Gold, CxG analysis, ODI, defensive profiles - all already built on top of the *old* unverified Silver output) get to treat its foundation as confirmed rather than assumed.

A repo audit today (21 Aug) found:
- `src/opponent_adjusted/pipelines/silver/builder.py` already contains ordered-upload logic (`# Enforce publication atomicity: data first, manifest next, completion marker last` — parquet files, then manifest.json, then `_SUCCESS`, uploaded in that explicit order). This looks like the intended fix already exists in code.
- But there is **no audit trail anywhere** (`docs/`, `audit_outputs/`) that this fix was ever exercised against a fresh publish, that `oam_core` was reconciled against it, or that the full test suite was ever run (only focused subsets like `tests/analysis/odi tests/features/cxg` have documented runs).
- 25 test files, 192 tests collected total as of this audit.
- No CI/GitHub Actions config exists — there has never been an automated full-suite gate.

This task is to close the gap for real, with evidence, not to assume the code fix is sufficient on its own.

## What to do

**1. Verify the fix is real, not just present.**
Read `builder.py`'s publish/upload logic end-to-end (the ordered-upload block, plus wherever `_SUCCESS` is checked/written) and confirm it actually guarantees `_SUCCESS` cannot be observed before all data+manifest objects are durably uploaded — trace the actual upload call sequence, don't just trust the comment. Report explicitly whether this holds.

**2. Run the FULL test suite — not a subset.**
`python -m pytest -q` (the whole suite, all 25 test files / ~192 tests, per `pyproject.toml`'s `testpaths = ["tests"]`). Report the real pass/fail count. If anything fails, stop and report — do not proceed to publish against a failing suite. This has never been done and documented before; it's the primary evidence gap.

**3. Publish a fresh compliant Silver output.**
Using the existing pinned `data_version` (`b0bc9f22dd77c206ddedc1d742893b3bbe64baec`) and current `silver_schema_version` (`statsbomb_silver_v1_2`, confirm this is still current in `contracts.py`), run the Silver builder to produce a new immutable Silver publication with the ordered-upload path. **Do not delete or overwrite the existing noncompliant prefix** — it stays as immutable evidence per the original remediation plan. If the builder's immutability/idempotency guard blocks republishing under the same `data_version`+`silver_schema_version` pair (check `_build_publication_plan`'s `existing > 0 and existing != expected` guard and any GCS-side prefix-exists check), report that constraint explicitly rather than working around it destructively — this may require a new versioned output path/prefix rather than overwriting, consistent with the project's "explicit immutable versioned paths" architecture rule (ADR-007). Do not guess at the right resolution if it's ambiguous — report back and ask.

**4. Reconcile `oam_core` against the new canonical Silver output.**
Run `publish_core.py`'s publication flow (or confirm via its existing row-count + join-check logic) against the freshly published Silver prefix. Report the row counts and join-check results it already computes (`shots_join_events_matches`, `three_sixty_frames_join_events_matches`, etc.) — these must reconcile cleanly, matching row-count expectations from the manifest.

**5. Write the closure report.**
`audit_outputs/silver_acceptance/silver_acceptance_closure_report.md` (new folder — none exists yet for this). Must include: full test suite pass/fail evidence with real counts, confirmation of the ordered-upload guarantee, the new compliant Silver prefix path (and confirmation the old noncompliant prefix was preserved, not deleted), and the `oam_core` reconciliation/join-check results. This is the first-ever documented full-suite run and the first-ever documented Silver acceptance closure for this repo — treat it as the canonical acceptance record.

## What NOT to do

- Do not delete or modify the existing (noncompliant) Silver prefix — it must remain as immutable evidence.
- Do not skip straight to declaring acceptance closed without the full-suite run and reconciliation evidence in hand.
- Do not touch ODI/defprofile code (`src/opponent_adjusted/analysis/{odi,defprofile}/`) or any CxG Gold feature code — this task is scoped to Silver/oam_core only.
- Do not silently work around an immutability guard by deleting/overwriting BigQuery rows or GCS objects — if the existing tooling won't let you publish a second compliant version cleanly, stop and report the exact blocker rather than improvising a destructive fix.
- Do not proceed to any further CxG/ODI/defprofile work until this closure report exists and is clean.

## Deliverables checklist

- [ ] Confirmation (with code trace, not just comment-reading) that the ordered-upload fix genuinely prevents a premature `_SUCCESS`
- [ ] Full test suite run: real pass/fail count, not a subset
- [ ] Fresh compliant Silver publication, old prefix preserved untouched
- [ ] `oam_core` reconciliation + join-check results against the new publication
- [ ] `audit_outputs/silver_acceptance/silver_acceptance_closure_report.md`

Report back with a summary when complete, or immediately if you hit the immutability/versioning ambiguity in step 3.
