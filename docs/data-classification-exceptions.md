# AI Data Classification Exceptions

[AGENTS.md](../AGENTS.md#data-classification) restricts AI tool usage to `public`
and `restricted` data. This register records each case where a user explicitly
released work involving `confidential` or `strictly confidential` data anyway, so
the decision stays traceable. It is a record, not an authorization: other
confidential data needs its own release and entry.

## Rules For This File

- **Describe the data, never reproduce it** — no names, records, values, keys, or
	excerpts. This file must itself stay within `restricted`.
- One entry per release, newest first, identifier `DC-NNNN`, written in the same
	change as the released work.
- Never widen an entry's scope; add a new one and link the old. When a release
	ends, set `Status: Withdrawn` or `Expired` and keep the entry.

## FOXES Data Boundary

FOXES source code and its published package data are public, but data supplied
to a calculation keeps its own classification. Wind-farm coordinates, turbine
and operational data, measured atmospheric series, customer/site identifiers,
and unpublished validation or research results can trigger the classification
question in [AGENTS.md](../AGENTS.md#data-classification). Do not infer that data
is public merely because FOXES can read its file format.

Use deterministic synthetic arrays or explicitly public packaged/example data
for tests and reproductions. A release for one data set does not cover another
site, customer, campaign, result, or derivative. If a software change itself
introduces processing of released confidential data, record that boundary in
[architecture.md](architecture.md#data-and-integration-boundaries) as required
by the repository policy.

## Status

No exceptions have been released.

## Entry Template

Copy this block above the previous entries when recording a release.

```markdown
### DC-NNNN: <short title of the feature, software, or data set>

- Status: Active | Expired | Withdrawn
- Date of release: YYYY-MM-DD
- Released by: <name and role of the person who explicitly released the work>
- Recorded by: <contributor or AI tool that performed the work>

**Covered.** Which feature, software, or data set, and what is explicitly not
covered.

**Why an AI tool.** What it was needed for, and which alternative was rejected
and why.

**Data categories.** The kinds of data involved — never the data itself.

**Protective measures.** What limits the exposure: anonymization, the AI tool and
deployment used, retention, access, deletion after the task.

**Follow-ups.** Open actions, review date, or when the release ends.

**Related records.** ADR, `docs/architecture.md` section, issue, or approval
thread.
```

## Related Records

- [AGENTS.md → Data Classification](../AGENTS.md#data-classification) — classes,
	triggers, procedure.
- [architecture.md](architecture.md) — classification of data the software
	itself processes.
