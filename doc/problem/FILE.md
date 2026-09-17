# VSF - AI Intern Projects 2026 (Batch 2, September)

Companion to the [July 2026 intern brief](dms-intern-projects-2026-en.md). That batch owns the
DMS user-guide RAG chatbot, AI-assisted Playwright automation for Sales and Aftersales, and the
English/Vietnamese string standardization tool. The four projects below are deliberately distinct
from those and are anchored in documented gaps across the DMS, STP, WMS and OM workspaces.
The [additional side-project bank](#additional-side-project-and-research-bank) expands the options
for unit testing, test coverage, AI reports, automation and optimization.

**Mentor for all projects:** Bùi Hữu Lộc ([locbh2@vingroup.net](mailto:locbh2@vingroup.net))


| Intern          | School | Interests                 | Primary project                               | Backup                             |
| --------------- | ------ | ------------------------- | --------------------------------------------- | ---------------------------------- |
| Trần An Thắng   | UET    | LLM, RAG                  | 1. STP technician assistant (RAG over TSBs)   | 5. Owner's Manual semantic search  |
| Trần Trung Hiếu | HUST   | Data, LLM, NLP            | 2. Warranty claim text mining                 | 8. V-Green contract extraction PoC |
| Trần Huy Hoàng  | UET    | Data, LLM, Infrastructure | 3. Application Insights incident triage agent | 7. Text-to-query assistant for BAs |
| Lê Trung Hiếu   | UET    | LLM, Infrastructure       | 4. Shared LLM evaluation and observability    | 9. DMS one-click deploy pipeline   |


Stretch goals for the second half of the internship: 6 (engineering runbook assistant) and
10 (AI code reviewer).

---



## 1. STP Technician Assistant - RAG over Technical Service Bulletins

**Intern:** Trần An Thắng

**Why:** STP already stores Technical Service Bulletins (TSBs), service documents and EPC assets in
MongoDB and the `vfstpblob` container, but technicians still search them by hand. A citation-backed
Vietnamese assistant that answers "how do I diagnose or repair X on model Y" is the cleanest RAG
problem in the estate, with a real corpus and a real user base.

### Deliverables

1. An ingestion pipeline from STP MongoDB and Blob into a vector store (reuse the `arkon`
  pgvector substrate rather than starting from zero).
2. A retrieval and answer API that returns Vietnamese answers with document citations and part
  numbers.
3. A golden set of 50-100 technician questions approved by the mentor or STP BA.
4. A side-panel proof of concept inside STP.FE.
5. Operations and knowledge-update documentation.



### Success Criteria

- At least 85% correct answers on the golden set, with citations, in under 5 seconds.
- Hybrid retrieval (keyword plus embedding) measured against pure embedding on the same set.
- End-to-end demo to the STP team and complete handover documentation.



### Plan



#### Phase 1

- Learn the STP domain, the TSB structure and the MongoDB/Blob layout.
- Learn RAG fundamentals and the `arkon` ingestion pipeline.
- Ingest one vehicle model's TSBs and answer questions over it.



#### Phase 2

- Full ingestion, chunking strategy for tables and part lists, citation formatting.
- Build the golden set, measure, iterate on retrieval and prompts.



#### Phase 3

- STP.FE side-panel PoC, prompt optimisation, documentation, demo and handover.

---



## 2. Warranty Claim Text Mining (WMS / FAMS)

**Intern:** Trần Trung Hiếu

**Why:** Warranty claims carry free-text failure descriptions (`VehicleWarrantyClaims` in the SQL
Server DataLake, plus `ErrorVin` notes) that nobody analyses systematically. Clustering and
classifying them exposes emerging defects weeks before they show up in aggregate counts.

### Deliverables

1. A cleaned, de-identified claim-text dataset with an agreed labelling scheme (failure area,
  component, symptom).
2. A classification pipeline (fine-tuned classifier or LLM labelling) with measured agreement
  against human coders.
3. A mapping from claim clusters to existing failure codes and TSBs.
4. A weekly "emerging defects" report per model, part and week.
5. Documentation of the data pipeline and how to re-run it.



### Success Criteria

- Labelled set of at least 2,000 claims with inter-annotator agreement documented.
- Classifier reaches at least 80% macro-F1 on a held-out set, or LLM labelling reaches at least
85% agreement with human coders.
- One emerging-defect signal confirmed as real by the warranty team.
- Report generation runs unattended.



### Plan



#### Phase 1

- Learn the warranty domain and the DataLake schema; agree the label scheme with the warranty BA.
- Extract, clean and de-identify the text; exploratory clustering.



#### Phase 2

- Label a seed set, build and evaluate the classifier or LLM pipeline, map clusters to TSBs.



#### Phase 3

- Emerging-defect report, automation, documentation, demo and handover.

---



## 3. Application Insights Incident Triage Agent

**Intern:** Trần Huy Hoàng

**Why:** DMS already has a KQL skill and a Teams alert skill under `DMS_DOCS/.cursor/skills/`,
but triage of new exceptions is still manual. An agent that clusters new exceptions, correlates
them with recent deploys and known incidents, and drafts a Jira ticket cuts time-to-first-triage.

### Deliverables

1. A scheduled job that pulls new exceptions from Application Insights and clusters them by
  signature.
2. Correlation with recent deploys and with existing incident and solution docs in DMS_DOCS.
3. Draft Jira VD tickets with a suspected root cause and linked evidence; Teams notification.
4. A precision/recall evaluation against a month of historical incidents.
5. Operations documentation and runbook.



### Success Criteria

- Agent identifies at least 80% of incidents that a human later filed, with fewer than 30% false
alarms, on the historical month.
- Measured reduction in time-to-first-triage on live DEV/UAT traffic.
- Drafted tickets are accepted by the team with minor edits.
- Runs unattended with a documented on-call procedure.



### Plan



#### Phase 1

- Learn the DMS estate, App Insights schema and the existing KQL and Teams skills.
- Build exception clustering over historical data.



#### Phase 2

- Add deploy and incident-doc correlation, LLM root-cause drafting, Jira and Teams integration.



#### Phase 3

- Evaluate on the historical month, run live in DEV/UAT, document and hand over.

---



## 4. Shared LLM Evaluation and Observability Platform

**Intern:** Lê Trung Hiếu

**Why:** Every AI project here (the user-guide chatbot, `arkon`, the standardization tool and the
three projects above) needs the same things: golden sets, LLM-as-judge scoring, prompt versioning,
cost and latency tracking, and regression runs in CI. Building it once makes all the other work
measurable and is a natural infrastructure project.

### Deliverables

1. A small service and SDK for registering prompts, golden sets and evaluation runs.
2. LLM-as-judge and exact-match scorers, with cost and latency capture per run.
3. Dashboards for accuracy, cost and latency over time per project.
4. CI integration so that a prompt or retrieval change fails the pipeline on regression.
5. Onboarding guide; at least two of the other intern projects integrated.



### Success Criteria

- The user-guide chatbot's existing golden suite runs through the platform with identical scores.
- At least two batch-2 projects report their metrics through it.
- A regression is caught in CI at least once during the internship.
- Documentation lets a new project integrate in under a day.



### Plan



#### Phase 1

- Survey the existing eval assets (`vinfast_chatbot` golden suite, `arkon` verify step).
- Stand up the service, prompt registry and first scorer.



#### Phase 2

- Dashboards, CI integration, LLM-as-judge, onboard the chatbot and one batch-2 project.



#### Phase 3

- Onboard remaining projects, harden hosting, document and hand over.

---



## Backup and stretch ideas

These were brainstormed alongside the four above and are kept here so they can be swapped in if a
primary project is blocked, or picked up in the second half of the internship.

1. **Owner's Manual semantic search for OM.** OM.FE has a full-text search API with no UI path to
  it (see `OM.DOCS/incidents/product-defects/2026-09-07-fe-search-unreachable.md`). Build a
   multilingual embedding search over published manual content, expose it from Laravel and wire a
   React search UI; compare BM25, embeddings and hybrid on a labelled query set.
2. **Engineering runbook assistant.** Extend `arkon` ingestion to DMS_DOCS, OM.DOCS, STP.DOCS and
  WMS.DOCS (about 2.3M words of runbooks and incidents), add role scoping and an evaluation
   harness. Stretch: let it invoke the existing `.cursor/skills` as tools.
3. **Text-to-query assistant for BAs.** Vietnamese questions to read-only FetchXML or WMS SQL,
  with schema RAG over the generated table catalog in `docs/power-apps/tables/uat65/`, PII masking
   and row limits; evaluated on 50 real BA questions.
4. **V-Green contract extraction PoC.** Feasibility is already scoped in
  `docs/solutions/vgreen-contract-extraction-feasibility-2026-09-10.md`; the blocker is Dataroom
   access, not technology. Build the OCR, LLM extraction, validation and Dataverse upsert pipeline
   on sample contracts so it is ready when connectivity lands.
5. **DMS one-click deploy pipeline.** The improvement plan records plugin and AppService deploys
  as fully manual. Wrap the existing deploy and verify skills and `Invoke-PacCached.ps1` in Azure
   DevOps pipelines, with an LLM step that drafts release notes and a post-deploy summary.
6. **AI code reviewer for GitLab MRs and Azure DevOps PRs.** Load each repo's `AGENTS.md`
  conventions as rules, post inline comments from CI, report precision on historical MRs, and
    restore a CI gate for DMS_DOCS.

---



## Additional side-project and research bank

**Brainstorm added:** 14 September 2026. These 25 proposals supplement the existing assignments;
they are candidate experiments, not confirmed implementation gaps or measured benefits.
"UniTest" is interpreted here as **unit testing**. "AI Report" covers both AI-written reports
and reports that evaluate AI systems.

Each MVP assumes one intern, one module or workflow, an available mentor and accessible sample
data. **S** means an estimated 1-2 focused weeks; **M** means 3-4 focused weeks, excluding access
delays and alongside-project scheduling. These are planning estimates, not delivery commitments.

### 1. Unit testing: create tests that catch meaningful defects

Use one authoritative C# module or JavaScript webresource as the pilot. Prefer existing test
infrastructure; confirm framework compatibility before selecting a generator or runner.

#### UT1. AI unit-test authoring assistant (M)

- **MVP:** Given one function, its requirements and nearby tests, draft normal, boundary and
failure cases with a reviewable test patch.
- **Research:** Does adding requirements and existing test examples improve results over a
source-only prompt?
- **Evaluate:** Compilation rate, mentor-accepted assertions, review time and detection of held-out
defects. Keep expected results independent of the implementation being tested.



#### UT2. Historical bug to regression-test converter (M)

- **MVP:** Turn five resolved DMS bugs with reproducible inputs into tests that fail before the fix
and pass after it. Deliver the paired revisions and a replay command.
- **Research:** Compare generation from a ticket alone with generation from the ticket plus a
sanitized trace. Reserve the fix for evaluation to avoid giving away the answer.
- **Evaluate:** Confirmed fail-before/pass-after cases, setup effort and misleading tests rejected.
If historical revisions cannot run locally, use explicitly labelled seeded defects.



#### UT3. Business-rule property testing lab (S)

- **MVP:** Generate edge cases for one agreed rule, such as line-total calculation, rounding or allowed status transitions. Preserve failing seeds and minimal counterexamples.
- **Research:** Do generated inputs reveal cases missed by hand-written examples under the same
runtime budget?
- **Evaluate:** Distinct confirmed failures, reproducibility and counterexample size. Have the BA
approve invariants, including rounding rules, before treating an output as incorrect.



#### UT4. Dataverse plugin fixture builder (M)

- **MVP:** Build reusable synthetic Target, pre-image, post-image and service-response fixtures for
one plugin handler; cover absent fields, null values and dependency failures.
- **Research:** Compare manual fixtures with schema-assisted generation for setup time and realism.
- **Evaluate:** Fixture reuse, mentor corrections and defects detected. Record platform behavior
the local doubles cannot represent and nominate separate integration checks for it.



#### UT5. Test assertion quality reviewer (S)

- **MVP:** Flag tests with no meaningful assertions, broad exception handling, excessive mocking
or assertions that simply copy the production calculation. Produce a local review report.
- **Research:** Compare static rules with an LLM reviewer on a mentor-labelled sample.
- **Evaluate:** Precision, missed weak tests and review time; include strong tests as negative
examples so the tool is not rewarded for flagging everything.



### 2. Test coverage: identify untested behavior and weak protection

Code coverage records executed lines, branches or methods. It is useful evidence, but assertion
quality needs a separate check. See Microsoft's
[code coverage guide](https://learn.microsoft.com/en-us/dotnet/core/testing/unit-testing-code-coverage).

#### TC1. Coverage report for changed code (S)

- **MVP:** Combine an existing coverage export with a Git diff to list changed executable lines
and branches that tests did not exercise. Start with one C# project.
- **Research:** Does a changed-code report help reviewers find relevant gaps faster than a
repository-wide percentage?
- **Evaluate:** Mapping accuracy on sampled diffs and reviewer time. Show missing instrumentation
as unknown, with explicit exclusions and denominators.



#### TC2. Mutation testing pilot (M)

- **MVP:** Introduce small code mutations in one business-rule module and inspect the surviving
mutations; add tests for a few mentor-confirmed gaps.
- **Research:** Which tests execute the code but fail to detect changed behavior? This is the
core experiment described in the [Stryker introduction](https://stryker-mutator.io/docs/).
- **Evaluate:** Detected mutations before and after, runtime and actionable survivors. Report
equivalent mutations, invalid mutations and timeouts separately.



#### TC3. Business-rule coverage map (M)

- **MVP:** Map one workflow's requirements to executable tests and their last results, for example
Work Order validation rules across status, role and input combinations.
- **Research:** Can an LLM suggest accurate requirement-to-test links using test names and bodies?
- **Evaluate:** Mentor-confirmed link precision and recall, plus requirements with no verified
check. A passing nearby test does not automatically cover a requirement.



#### TC4. Test-gap prioritizer (M)

- **MVP:** Rank ten test candidates using uncovered branches, recent change frequency and known
incident history, with the reasons visible for each ranking.
- **Research:** Does this ranking find more useful tests than sorting only by lowest coverage?
- **Evaluate:** Mentor relevance ratings and confirmed defects found for equal engineering effort.
Use older incidents to build the ranking and later incidents to assess it.



#### TC5. Integration failure coverage matrix (M)

- **MVP:** Exercise one local callback or queue handler with duplicate, delayed, malformed and
out-of-order messages, timeouts and partial dependency failures.
- **Research:** Which fault combinations reveal gaps that ordinary success-path tests miss?
- **Evaluate:** Correct outcomes against an agreed contract, duplicate side effects and missing
recovery checks. Report scenario coverage separately from line coverage.



### 3. AI reports: turn evidence into useful, checkable explanations

For these projects, calculate figures with deterministic code and give the AI the resulting
tables plus evidence identifiers. Compare the output with a plain report template. Reuse project
4's evaluation platform if available; a local dataset and runner are enough to start.

#### AR1. AI quality and test-run digest (S)

- **MVP:** Turn one test run, coverage export and change list into a short report: what failed,
what changed, what remains untested and the next investigation steps.
- **Research:** Does AI explanation help developers diagnose results faster than a template?
- **Evaluate:** Factual accuracy, evidence-link correctness, unsupported claims and reader task time.
Identify the source revision and test run on every report.



#### AR2. Business KPI narrative assistant (M)

- **MVP:** Explain a fixed weekly dataset for one topic, such as aging Work Orders or warranty
turnaround, with drill-down references and a glossary of metric definitions.
- **Research:** Compare a template with an LLM narrative for usefulness and numerical fidelity.
- **Evaluate:** Recomputed-number agreement, BA usefulness scores and unsupported causal claims.
Start with a sanitized export; a KPI shift alone does not establish its cause.



#### AR3. Release verification evidence report (M)

- **MVP:** Convert saved source comparisons, package manifests, deployment results and smoke-test
outputs into an evidence-linked release summary with pass, fail and unknown states.
- **Research:** Can the report correctly distinguish source changes, deployed components and
runtime behavior when some evidence is deliberately missing?
- **Evaluate:** Claim-to-evidence accuracy and false-ready conclusions on mentor-labelled bundles.
This complements project 9 by consuming artifacts rather than performing deployments.



#### AR4. AI report fact-checker (M)

- **MVP:** Check generated reports against their input tables and artifacts; flag wrong totals,
unsupported comparisons, missing citations and comparisons across incompatible time windows.
- **Research:** Compare deterministic validation, an LLM critic and their combination.
- **Evaluate:** Precision and recall on reports containing planted errors, plus false alarms on
correct reports. Keep the evaluation examples separate from prompt-tuning examples.



#### AR5. AI experiment comparison report (S)

- **MVP:** Generate a comparison of two prompt or retrieval versions: quality, response time,
measured cost and representative failures on the same questions.
- **Research:** Does presenting paired examples change which version reviewers select compared
with showing averages alone?
- **Evaluate:** Reproducible calculations, reviewer agreement and correct identification of
tradeoffs. Extend project 4's reporting rather than creating another evaluation platform.



### 4. Automation: remove a repeatable engineering chore

These ideas support the July batch's Playwright work through fixtures, diagnosis and tooling;
they do not create a second Sales/Aftersales browser-test project.

#### AU1. Synthetic test-data factory (M)

- **MVP:** Generate a deterministic local fixture bundle for one workflow, including valid related
records and deliberately invalid cases. Include a seed, manifest and replay command.
- **Research:** Compare schema-only generation with schema plus BA-defined business constraints.
- **Evaluate:** Constraint validity, scenario diversity and developer setup time. Any later DEV/UAT
seeding needs an explicit environment and ownership-based cleanup plan.



#### AU2. CI failure classifier (S)

- **MVP:** Classify saved failed-job logs as compilation, test assertion, dependency, authentication
or infrastructure failures and link to the relevant runbook.
- **Research:** Compare keyword rules, text similarity and an LLM on labelled historical logs.
- **Evaluate:** Per-category precision/recall, abstentions and triage time. Keep this focused on CI
failures to complement project 3's application-incident triage.



#### AU3. Documentation and contract drift detector (M)

- **MVP:** Compare one API contract or configuration schema with its documentation and draft a
report of removed fields, changed requirements or missing examples.
- **Research:** Compare structured diffs with LLM interpretation for actionable drift detection.
- **Evaluate:** Precision on historical changes, missed incompatibilities and review effort.
Begin with one contract type and generate a patch only when the evidence is sufficient.



#### AU4. Developer setup doctor (S)

- **MVP:** Check one repository's SDK, dependencies and local service prerequisites and output
exact remediation steps with a machine-readable result.
- **Research:** Does dependency-aware diagnosis explain setup failures better than a flat checklist?
- **Evaluate:** Diagnosis accuracy for seeded setup problems and time to a successful local build.
Make checks repeatable without silently altering developer configuration.



#### AU5. Flaky-test reproducer (M)

- **MVP:** Run selected suspect tests with controlled seeds, order and parallelism; bundle the
smallest reproducible failure with its logs and environment details.
- **Research:** Which perturbations expose order dependencies, timing problems or shared state?
- **Evaluate:** Reproduction rate per runtime budget and confirmed causes. Reruns should preserve
the first failure as evidence instead of turning a later pass into a clean result.



### 5. Optimization: measure a bottleneck and test a bounded improvement

Choose a reproducible local workload first. Record revision, data size, repetitions, warm/cold
state, median and tail latency, errors and output correctness. Local results describe that workload;
they do not by themselves establish a production improvement.

#### OP1. FetchXML performance experiment assistant (M)

- **MVP:** Extend the existing entity-list bisect workflow with an experiment manifest and report
for one slow view. Vary one factor at a time: columns, sorting, joins or total counts.
- **Research:** Which factors explain the cost, and which faster variants preserve the required
records, ordering and user-visible behavior?
- **Evaluate:** Repeated timings and semantic equivalence. Label behavior-changing variants as
diagnostics; follow Microsoft's [FetchXML performance guidance](https://learn.microsoft.com/en-us/power-apps/developer/data-platform/fetchxml/optimize-performance),
including its restrictions on query hints.



#### OP2. Duplicate API-call detector (S)

- **MVP:** Analyze one saved HAR or instrumented local page flow, group potentially redundant
requests and show their initiators and timing in a waterfall report.
- **Research:** Can request signatures plus timing distinguish accidental duplicates from valid
refreshes, retries and pagination?
- **Evaluate:** Mentor-confirmed duplicate precision; then compare request count, bytes and task
completion time after one fix while checking freshness and error behavior.



#### OP3. Plugin and API hotspot benchmark (M)

- **MVP:** Profile one locally reproducible handler and compare one proposed improvement, such as
retrieving fewer fields or avoiding repeated lookups, in the existing local harness.
- **Research:** Does time concentrate in computation, serialization or dependency calls?
- **Evaluate:** Latency, allocations where measurable, dependency-call count and identical business
outcomes. Keep real dependency timings distinct from simulated ones.



#### OP4. Cache strategy simulator (M)

- **MVP:** Replay a synthetic or sanitized request/update trace against no-cache, fixed-expiry and
explicit-invalidation strategies for one lookup workload.
- **Research:** How do hit rate, backend load and stale responses change as update frequency rises?
- **Evaluate:** Hit rate, latency, memory and stale-response duration, including cold starts and
failed refreshes. Validate behavior against a defined freshness requirement.



#### OP5. LLM cost, latency and quality experiment (M)

- **MVP:** Compare prompt trimming, context selection and a simple routing policy on one existing AI project's fixed evaluation set.
- **Research:** Which configuration reduces measured cost or latency while retaining acceptable
answer quality, including difficult and unanswerable questions?
- **Evaluate:** Paired quality scores, tail latency, tokens and cost using recorded pricing and
dates. Reuse project 4's instrumentation; hold out the final evaluation set.



### Suggested starting shortlist

This shortlist is a planning judgement based on MVP scope and likely input availability. Confirm
the selected module, available fixtures and mentor capacity before assigning work.


| Area          | First choice                     | Why start here                                           | Research alternative       |
| ------------- | -------------------------------- | -------------------------------------------------------- | -------------------------- |
| Unit testing  | UT2: Bug to regression test      | Concrete fail-before/pass-after demonstration            | UT3: Property testing      |
| Test coverage | TC1: Coverage for changed code   | Small output reviewers can use immediately               | TC2: Mutation testing      |
| AI reports    | AR1: Quality and test-run digest | Starts from saved artifacts with verifiable facts        | AR4: Report fact-checker   |
| Automation    | AU2: CI failure classifier       | Offline baseline and straightforward labelled evaluation | AU5: Flaky-test reproducer |
| Optimization  | OP2: Duplicate API-call detector | One captured workflow gives a bounded investigation      | OP1: FetchXML experiments  |


Optional pairings with current interests: Trần An Thắng can explore AR4 or OP5; Trần Trung Hiếu
can explore AR2 or TC4; Trần Huy Hoàng can explore AU2 or OP3; Lê Trung Hiếu can explore TC1 or
AU4. These are discussion options, not changes to the primary assignments.

For a shared demo, connect **UT2 → TC1 → AR1**: reproduce a historical bug, show what the test
covers, then explain the run with linked evidence. Keep each component usable independently.

### Common experiment and handover format

1. Name one user, one workflow, the problem and the baseline to beat. Confirm access during the
  first working session; use local or synthetic inputs when live access is unnecessary.
2. Deliver a reproducible MVP before adding a UI or agent orchestration. Store the revision,
  configuration, input manifest and evaluation command with the results.
3. Compare against a simple baseline on the same held-out cases and effort/runtime budget.
  Report sample sizes, failures and limitations; an experiment finding no improvement is useful.
4. Demo the result and hand over setup steps, reviewed findings and a bounded next increment.
  Keep durable DMS notes in DMS_DOCS and bulk generated evidence in DMS_FILES.



### Workspace impact of this brainstorm

The current `DMS.code-workspace` was checked for membership. This session expands this document
only; candidate implementation homes below are not source audits or implementation assignments.


| Workspace repository       | Classification for this session | Relevance to later implementation                                                |
| -------------------------- | ------------------------------- | -------------------------------------------------------------------------------- |
| DMS_DOCS                   | Affected                        | Project brief, experiment descriptions and future durable handovers              |
| DMS_PROD                   | Not applicable                  | Candidate plugin and webresource tests or fixes after project selection          |
| VinFast.DMS.Api            | Not applicable                  | Candidate API, callback and queue experiments                                    |
| VF.TNS.Integration         | Not applicable                  | Candidate legacy integration tests; verify local build feasibility first         |
| VinFast.AppService.NetCore | Not applicable                  | Candidate inbound API contract and performance experiments                       |
| Technosoft.Yana.Vinfast    | Not applicable                  | Candidate legacy rule tests; verify source ownership and build feasibility first |
| DMS_LOCAL_HOST             | Not applicable                  | Candidate local replay and benchmark harness reuse                               |
| bu-setup-automation        | Not applicable                  | No provisioning implementation requested in this brainstorm                      |


No sibling-repository change is needed to define these options. STP, WMS and OM remain contextual
examples from the original brief, outside the current workspace membership. Microsoft sample
repositories remain reference-only, and the decompiled `technosoft-dms` archive remains read-only.
 