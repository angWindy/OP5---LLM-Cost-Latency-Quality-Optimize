# Engineering review and improvement roadmap

- Review date: 26 September 2026.
- Repository: [op5-llm-perf](https://gitlab.vinsmartfuture.tech/vsf-vf-kdhmdl/intern/thangta3/op5-llm-perf).
- Reviewed branch: `main` at `2a54f37978dcfcadc8e362c3279902682f5c1a21`.
- Review scope: source, documented requirements, available tests and CI configuration at this revision. This report adds feedback only; it does not implement the recommendations.
- Audience: the intern maintaining this repository and their mentor. Effort estimates are approximate implementation effort, not an assessment of the intern.
- Priorities: P1 = address before relying on the affected capability; P2 = next engineering increment; P3 = follow-up improvement. S = up to one day; M = roughly two to three days.

## Current state and evidence

The reviewed commit contains exactly one tracked file: [README.md:1](https://gitlab.vinsmartfuture.tech/vsf-vf-kdhmdl/intern/thangta3/op5-llm-perf/-/blob/2a54f37978dcfcadc8e362c3279902682f5c1a21/README.md#L1). It is the GitLab starter template. There is no implementation, dependency manifest, automated test suite or repository CI configuration to validate. The project exists and provides a place to collaborate, but application behavior cannot yet be assessed. This is a repository readiness assessment; absence of source here does not establish whether the intern has work elsewhere.

## Findings and next increment

1. **P2 — Define what is being compared (S).** The README contains no workload, baseline, model/provider configuration or quality requirement. A proposed MVP is a reproducible comparison of two inference configurations on the same small, public or synthetic prompt set. The repository name suggests performance work; the exact assignment and allowable providers still need mentor agreement.
2. **P2 — Design a benchmark that preserves comparability (M).** Record dataset hash, requested and served model version, parameters, concurrency, warm-up policy and tool version. Measure total latency, time to first token only when streaming permits it, throughput, error/timeout counts and token usage. Acceptance: an offline fake provider produces deterministic metric calculations; failures remain in denominators; unsupported measurements are reported as unavailable.
3. **P2 — Pair speed with quality and cost evidence (M).** Predefine a task-specific quality rubric and a fixed evaluation set before selecting a faster configuration. Acceptance: repeated runs report sample counts and latency distributions, the same workload is used for each arm, and cost is either recorded from provider usage or clearly labelled as an estimate with its rate/date. No paid run is required for the first harness milestone.

## Suggested handoff

Start with an offline harness, two fake response profiles, JSON results and a short generated comparison. Specify the live-run budget and stop conditions with the mentor before using paid services. Add a GitLab MR test job for metric calculations. Avoid declaring a configuration faster from a single request or changing prompt quality while comparing latency.

## Validation and limits

Verified the remote default branch, commit identity and tracked file inventory, and inspected the starter README. Application tests, builds, runtime behavior and deployment checks are not applicable because no application is present at this revision. The report's links and whitespace are checked before publication. There were no open GitLab issues or MRs at initial review discovery.
