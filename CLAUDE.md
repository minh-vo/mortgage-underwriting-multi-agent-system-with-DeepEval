# CLAUDE.md: build brief for the Mortgage Underwriting Multi-Agent System

This file hands the build to a coding agent. Read all of it before writing code.
**Status:** build steps 0-7 are written and `uv run pytest` is green (106 tests). No live Claude call has been made yet. Still open: the approved live runs, `eval/results.json`, the README results table and `demo.gif`. See section 14 for where the code differs from this brief.

---

## 1. Goal

Build a small, readable Python program for a public GitHub portfolio (`minh-vo/mortgage-underwriting-multi-agent-system-with-DeepEval`). An employer sees a screen recording on the GitHub page. They can then clone the repo, add an Anthropic API key, and watch a team of AI agents underwrite a mortgage application in a local Streamlit page. There is no hosted demo.

The program re-implements the architecture of a course notebook the owner completed:
- four specialist analysts (credit, income, assets, collateral);
- a Supervisor;
- a Critic exposed over A2A;
- a Decision Agent;
- human-in-the-loop review;
- deterministic calculators served over MCP;
- DeepEval evaluation.

It also **fixes the design problems** that notebook had (section 3).

**Guiding principle: simple and straightforward.** Prefer fewer files, plain functions and obvious data flow over abstractions. Every component must be explainable in two sentences in the README.

---

## 2. Decisions already made (do not revisit)

| Topic | Decision |
|---|---|
| Language | Python 3.12, managed with `uv` (`uv` is installed at `~/.local/bin/uv`) |
| LLM | **Claude via the official `anthropic` Python SDK.** No LangChain model wrappers, no OpenAI, no OpenAI-compatible shims. |
| Default model | `claude-opus-5-5` for every agent; configurable via `.env` (`MODEL`, `JUDGE_MODEL`) |
| Orchestration | LangGraph (graph, `Send` fan-out, `interrupt()`, SQLite checkpointer) |
| Tools | MCP server using the official `mcp` SDK (FastMCP), **stdio transport** |
| Critic | Separate **A2A** service process (official `a2a-sdk`), port 9105 |
| UI | Streamlit (`app.py`), run locally (section 8) |
| Distribution | GitHub repo only. No web hosting, no hosted demo page, no sign-in flow. The README carries a screen recording (`demo.gif`). |
| Policy retrieval | Section lookup and keyword search over a markdown policy file. **No vector DB.** |
| Evaluation | DeepEval, with a judge built on the `anthropic` SDK |
| Data | Policy text and synthetic cases written for this repo. The course files are never reused; their content may be paraphrased (section 6). |

**Dependency versions: stay on the versions the course notebook pins.**

| Package | Pin | Source |
|---|---|---|
| `langgraph` | `==1.1.9` | notebook |
| `mcp[cli]` | `==1.30.0` | notebook |
| `a2a-sdk` | `==1.1.2` | notebook |
| `deepeval` | `==4.2.2` | notebook |
| `anthropic`, `langgraph-checkpoint-sqlite`, `aiosqlite`, `streamlit`, `pydantic`, `python-dotenv`, `pytest` | the newest version `uv` resolves **alongside the four pins above**, then pinned exactly | not in the notebook |

The four notebook pins are never bumped. If `uv` can't resolve a newer package against them, lower the newer package; if that still fails, stop and report. The notebook's other packages (`langchain*`, `chromadb`, `pypdf`, `langchain-mcp-adapters`, `nest_asyncio`) aren't used and aren't installed.

**Before writing any Claude API code, load the `claude-api` skill** and read `python/claude-api/README.md` and `python/claude-api/tool-use.md`. Do not guess SDK signatures. Several shapes changed recently: adaptive thinking, `output_config.effort`, structured outputs, server-side refusal fallbacks, and preserved thinking blocks in tool loops.

---

## 3. Problems in the old notebook, and the required fix for each

These requirements drive the design. The README will present them neutrally as "design choices in this version"; don't criticise the course.

| # | Problem observed | Required fix |
|---|---|---|
| P1 | MCP server and A2A service ran as threads inside one notebook process, with model and HTTP clients shared across event loops, causing A2A timeouts | MCP runs as a **stdio subprocess**. The Critic runs as **its own process**. **No client is ever shared across event loops:** each run opens its own `AsyncAnthropic` client and closes it when the run ends (section 7.2). All network calls set explicit timeouts, and a caller's timeout is always longer than the callee's worst case. |
| P2 | The Supervisor LLM produced a plan once, and routing was hard-coded, so the plan was decorative and Plan Adherence scored 0 | The Supervisor chooses the **next step on every pass**, from the moves code allows (section 7.3). What it picks is what runs, and each pick is logged with a reason and the moves it could have picked. |
| P3 | A reworked specialist never saw the Critic's feedback, so it repeated itself until the rework cap | State field `rework_feedback[area]`. A reworked specialist gets the Critic's defect text **and its own previous analysis**. |
| P4 | The Critic failed analyses for disagreeing with their conclusion (co-underwriting instead of reviewing) | The Critic returns `defects` (problems with the analysis) **separately** from `application_risks` (weaknesses of the borrower). **The verdict is computed in code:** fail if and only if there is at least one `major` defect. |
| P5 | When a specialist hit its tool-round cap, a placeholder string was stored and treated as a finished analysis | Each analysis is a structured object with `status: "complete" \| "failed"`. A failed analysis is treated as missing. After 2 failed attempts the area is not retried and the case goes to human review. |
| P6 | Reserve months went negative when the borrower couldn't fund closing (reported as "Reserves -7.9mo") | `metrics.py` returns `funds_shortfall` (≥ 0), and `reserve_months = 0` when closing can't be funded. |
| P7 | Human review triggered only on CONDITIONAL_APPROVAL, so a 1-point score change removed the demo | Review triggers on any of: (1) decision CONDITIONAL_APPROVAL; (2) risk score within ±5 of a band edge (26–35, 56–65); (3) the Critic's last verdict is still fail when the Decision is made; (4) any analysis is missing or failed. |
| P8 | Specialists ran one after another, although they are independent | Specialists selected in one Supervisor pass run **in parallel** via LangGraph `Send`. |
| P9 | `MemorySaver` lost paused cases on restart | `AsyncSqliteSaver` (`langgraph-checkpoint-sqlite`), file `underwriter.db` (git-ignored). The async saver is required because the graph nodes are async. |
| P10 | DeepEval reference arguments disagreed with the real tool inputs; argument values were never compared; the judge was the same small model | Reference arguments come from the **same** `metrics.py` functions the agents use. `ToolCorrectnessMetric(evaluation_params=[ToolCallParams.INPUT_PARAMETERS], available_tools=...)`. The judge model is configured separately (`JUDGE_MODEL`). |
| P11 | An expected label (CONDITIONAL) conflicted with the written policy, and case records carried pre-computed figures that disagreed with their own raw numbers | Case records hold raw facts only; every figure is computed in `metrics.py`. Labels in `data/cases.json` must follow `data/policies.md`. The eval accepts a **set** of acceptable outcomes per case, because LLM outcomes are probabilistic. |

---

## 4. Hard rules

1. **Never commit course material or secrets.** The first action is to update `.gitignore` (section 5). `docs/` holds the course PDF, notebooks and `config.json` with **live API keys**. Never read `docs/config.json`, and never print or log any key.
2. **No git push and no publishing of any kind** without the owner's explicit go-ahead. Local commits only when asked.
3. **Calls that cost money need approval.** Unit tests must run with no API key. Before any live Claude call (smoke test, Streamlit run, eval), ask the owner, then report actual token usage (`response.usage`) after the first run.
4. Keep `docs/` and `backup/` untouched; they aren't part of the program. Reading `docs/underwriting_policies.pdf` and `docs/mortgage_test_cases.json` is allowed for the paraphrasing work and the originality check in section 6, and for nothing else.
5. When writing code for `claude-opus-5-5`, enable the server-side refusal fallback by default: `client.beta.messages.create(..., betas=["server-side-fallback-2026-07-01"], fallbacks="default")`. Tell the owner it's on. Re-verify the call shape in the skill docs before use.

---

## 5. Repository layout (target)

```
.gitignore              # add: docs/  backup/  .env  *.db  .deepeval/  .claude/  __pycache__/  .venv/  .pytest_cache/
.env.example            # ANTHROPIC_API_KEY=   MODEL=claude-opus-5-5   JUDGE_MODEL=claude-opus-5-5   CRITIC_URL=http://127.0.0.1:9105
pyproject.toml          # uv project; package "underwriter" under src/; exact pins (section 2)
README.md               # rewritten for this program (section 11)
CLAUDE.md               # this file
demo.gif                # screen recording shown in the README (added at build step 8)
data/
  policies.md           # policy, numbered sections (section 6.1)
  cases.json            # 4 synthetic applications (section 6.2)
src/underwriter/
  __init__.py
  config.py             # loads .env (python-dotenv); Settings dataclass; MAX_REWORKS; model price table
  models.py             # all Pydantic models / TypedDicts shared across modules
  metrics.py            # every calculation and threshold; single source of truth
  policy.py             # parse policies.md -> {section_id: text}; get_section(), search(query)
  sanitize.py           # PII masking
  bias.py               # protected-characteristic keyword screen
  mcp_server.py         # FastMCP stdio server (section 7.1)
  mcp_client.py         # open stdio session, list tools as Anthropic tool defs, call tool
  llm.py                # Anthropic helpers: use_client(), structured(), tool_loop() (section 7.2)
  agents/
    __init__.py
    specialists.py      # one runner + four prompt builders
    critic.py           # review(), compute_verdict()
    decision.py         # decide(), band(), needs_human_review(), validate_human_review()
    supervisor.py       # allowed_moves(), choose_next()
  critic_service.py     # A2A server wrapping agents/critic.py (python -m underwriter.critic_service)
  a2a_client.py         # ensure_critic_running(), send review request to the Critic service
  graph.py              # LangGraph assembly; run_case(), resume_case() (section 7.3)
  fake_llm.py           # deterministic stand-in for tests (no network)
app.py                  # Streamlit UI (section 8)
eval/run_eval.py        # DeepEval runner (section 9)
eval/results.json       # results of the last approved eval run; committed, shown in the app's Evaluation tab
tests/
  test_metrics.py  test_policy.py  test_sanitize.py  test_mcp_server.py
  test_supervisor.py  test_critic_verdict.py  test_decision.py  test_graph.py  test_app.py
```

---

## 6. Data: paraphrase the course content, never reuse the files

The course gave a 14-page policy PDF and a JSON file of three test cases. Neither file, nor any extract of one, goes into the repo. Their **content may be paraphrased**.

Every page of the PDF carries a notice against publishing its contents in part or in full, so paraphrase at the level of **rules and subject matter**, not sentence by sentence. The rules themselves are ordinary industry conventions (620 minimum score, 28/43 ratios, a 7-year foreclosure wait). Use our own structure, our own wording and, where listed below, our own thresholds. Carry over no sentence, heading wording, name, identifier or footer text. Write from the rule lists below, not with the PDF open alongside.

### 6.1 `data/policies.md`

**Scope:** conventional loan, primary residence, single-unit purchase. Leave out the original's FHA/VA/USDA/jumbo, second-home, investment, rural, flood and documentation-type material. That is what takes 14 pages down to about 2.

**Format:** the first line says this is a synthetic policy written for a demo and is not lending guidance. Each section is a `## 1.1 Title` heading (so `policy.py` can split on them), one or two plain sentences, then a short bullet list of thresholds.

**Rules.** `metrics.py` and the tests depend on them:

- **1 Credit**
  - 1.1 Minimum score 620. Tiers: 760+ Excellent, 700–759 Good, 660–699 Fair, 620–659 Marginal (needs 2 compensating factors from 2.5).
  - 1.2 Bankruptcy waiting period from discharge: Chapter 7 = 4 years, Chapter 13 = 2 years.
  - 1.3 Foreclosure waiting period 7 years.
  - 1.4 Late payments: no 60+ day lates in the last 12 months; up to two 30-day lates in 12 months are allowed with a letter of explanation (LOE).
  - 1.5 Collections: non-medical collections over $2,000 in total must be paid before closing; medical collections are ignored.
- **2 Income**
  - 2.1 Two years in the same line of work.
  - 2.2 Employment gaps over 30 days in the last 2 years need an LOE. A gap over 6 months requires 6 months on the current job.
  - 2.3 Self-employed: 2-year average of net income. If income declined more than 20% year over year, use the lower recent year and flag it.
  - 2.4 Bonus income counts only if received for 2 years and expected to continue; use the 2-year average. Otherwise exclude it.
  - 2.5 Ratios: front-end ≤ 28%; back-end ≤ 43%. Back-end up to 50% is allowed with **at least 2** compensating factors: credit score ≥ 720; post-closing reserves ≥ 12 months; LTV ≤ 75%. A front-end ratio above 28% is a noted risk and needs the same two compensating factors. Back-end above 50% is not allowed.
- **3 Assets**
  - 3.1 Required reserves: 2 months PITI, or 6 months if back-end DTI > 43%, the borrower is self-employed, or the credit score is < 680.
  - 3.2 Down payment and closing costs from verified liquid funds (checking + savings) or a documented gift.
  - 3.3 Any non-payroll deposit over 50% of monthly qualifying income needs source documentation. Until documented, it is deducted from verified funds.
  - 3.4 Retirement balances don't count toward closing funds or reserves.
- **4 Collateral**
  - 4.1 LTV is measured on the lower of purchase price and appraised value.
  - 4.2 Health, safety or structural repairs must be completed before closing. Cosmetic repairs up to $5,000 may use an escrow holdback.
  - 4.3 Maximum LTV 95% (primary residence). Above 80% requires mortgage insurance; note this, it is not a denial.
  - 4.4 Eligible types: single-family, townhouse, condominium.
  - 4.5 A condominium requires a project review.
- **5 Decision and review**
  - 5.1 Risk score bands: 0–30 approve, 31–60 conditional approval, 61–100 deny.
  - 5.2 A human underwriter reviews the case when any P7 trigger applies (list the four).
  - 5.3 A human decision that differs from the recommendation needs a written justification.

**Deliberate differences from the course policy** (ours are simpler and fully deterministic): the collections threshold; the gap rules; the self-employed decline rule; the large-deposit threshold; retirement funds excluded; a flat $5,000 holdback; the exception needing at least 2 of 3 named factors; section 5; and our own section numbering.

### 6.2 `data/cases.json`

Four applications (the course file has three). Cover the same ground as the course records (identity, credit history, employment, debts, assets, loan, property), restructured so every policy fact is a field, not free text.

**Record shape:**
- `case_id`, `application_date` (the same fixed date for all four, `2026-03-02`), a one-line `profile` for the UI.
- **Identity:** name, SSN, phone, email, street address, so the sanitizer has something to mask. All new: none of the course file's names, employers, addresses, emails or phone numbers. SSNs from the never-issued 900 range, phones in the reserved 555-01xx block, emails at `example.com`. Different occupations and cities from the course cases.
- **Credit:** `credit_score`; `bankruptcy` as `{chapter, discharge_date}` or null; `foreclosure_date` or null; `late_payments` as a list of `{days_late, date, account}`; `collections` as a list of `{type: medical | non_medical, amount}`; a short `notes` string.
- **Employment:** `type` (W2 | self_employed), `start_date`, years in the line of work, `gaps` as a list of `{start, end, reason, documented}`; base salary; bonus by year with an `expected_to_continue` flag; self-employed net income by year.
- **Debts:** a dict of monthly payments by name. No totals.
- **Assets:** checking, savings, retirement; `gift_funds {amount, documented}`; `recent_deposits` as a list of `{date, amount, description, payroll, source_documented}`.
- **Loan:** amount, down payment, closing costs, and PITI broken into principal-and-interest, taxes, insurance, HOA, MI.
- **Property:** purchase price, appraised value, type, `repairs` as a list of `{description, cost, category: structural | safety | cosmetic}`.
- **`expected`** (a sibling block, not part of the application): `acceptable_outcomes`, `human_review_expected`, `facts`. The loader strips it before the application reaches any agent, so a label can never leak into a prompt.

**Not in the record:** any pre-computed figure (DTI, reserve months, liquid total, debt total). `metrics.py` is the only place figures are computed (P11).

**Design targets.** Draft figures; `test_metrics.py` confirms them. If a figure doesn't produce the intended fact, change the data, not the rule.

| | CASE-001 | CASE-002 | CASE-003 | CASE-004 |
|---|---|---|---|---|
| Profile | Strong W-2 earner | Borderline, curable | Near the deny edge | Weak |
| Score | 775 | 728 | 668 | 588 (below minimum) |
| Qualifying income / mo | 12,125 (salary + 2-year average bonus) | 9,000 (one-year bonus excluded) | 7,500 | 5,833 (self-employed, down 27%, so the lower year) |
| PITI / other debts | 2,900 / 750 | 2,400 / 1,750 | 2,050 / 1,500 | 2,300 / 1,900 |
| Front / back ratio | 23.9% / 30.1% | 26.7% / 46.1% | 27.3% / 47.3% | 39.4% / 72.0% |
| Loan / LTV basis / LTV | 400k / 520k / 76.9% | 264k / 356k (appraisal below price) / 74.2% | 279k / 310k / 90% (MI) | 305k / 318k / 95.9% (over max) |
| Cash to close / verified liquid | 131k / 175k | 103.5k / 125k | 37.5k / 45.5k | 31k / 7k (19k less a 12k unsourced deposit) |
| Reserves (required) | 15.2 mo (2) | 9.0 mo (6) | 3.9 mo (6) | 0 mo, shortfall 24k (6) |
| DTI exception | not needed | allowed: score ≥ 720 and LTV ≤ 75% | not allowed: no compensating factors | over max |
| Other facts | no derogatories | one 30-day late 7 months ago (LOE); condo (project review); 3.2k cosmetic repairs (holdback) | 2-month job gap with an LOE on file | Chapter 13 discharged 8 months ago; a 60-day late; 2.6k non-medical collection; structural and safety repairs |
| Acceptable outcomes | {APPROVED} | {CONDITIONAL_APPROVAL} | {CONDITIONAL_APPROVAL, DENIED}; human review expected | {DENIED} |

### 6.3 Checks on the data

- `test_metrics.py`: the fact table above, per case.
- `test_policy.py`: every threshold constant in `metrics.py` appears in the matching section of `policies.md`, so policy text and code can't drift apart.
- **Originality check, run locally and never committed** (it needs `docs/`): no 8-word sequence shared between `policies.md` and the PDF text; no name, employer, address, email, phone or SSN from the course JSON in `cases.json`; the PDF's footer identifiers appear nowhere in tracked files. Report the result to the owner.
- **Owner review (checkpoint 1a):** show the owner the policy and the case table before building on them.

---

## 7. Components

### 7.1 `metrics.py` and the MCP server

**`metrics.py`:** pure functions, no I/O, fully unit-tested. All policy thresholds are named constants here. Every time period (waiting periods, "last 12 months", "last 2 years") is measured from the case's `application_date`; never call `date.today()`.

- **Helpers:**
  - `qualifying_income(app)` returns the total and a note (bonus and self-employed rules per 2.3–2.4).
  - `monthly_debts(app)`, `piti(app)`.
- **`case_metrics(app) -> dict`**, with keys:
  - `qualifying_income`, `existing_debt`, `piti`;
  - `front_end_ratio`, `front_end_over_limit`, `back_end_dti`;
  - `ltv_basis`, `ltv`, `ltv_over_max`, `mi_required`;
  - `cash_to_close` (down payment + closing costs);
  - `liquid_assets` (checking + savings + documented gift funds, minus unsourced large deposits);
  - `post_closing_liquid`, `funds_shortfall`, `reserve_months` (≥ 0), `required_reserve_months`;
  - `compensating_factors` (list), `dti_exception` (`"not_needed" | "allowed" | "not_allowed" | "over_max"`);
  - `large_deposits_unsourced` (list), `holdback_limit`;
  - `credit_tier`, `derogatory_flags` (list: waiting periods not met, late-payment breaches, collections to pay);
  - `repair_flags` (structural or safety repairs; cosmetic total vs the holdback limit), `property_type_eligible`, `project_review_required`.
- **`tool_inputs(app) -> dict[tool_name, dict]`** gives the exact arguments each calculator tool should receive. It's used **both** in specialist prompts ("authoritative tool inputs") and as DeepEval reference arguments (fix P10).
- **Calculator functions** with typed signatures and docstrings. They become MCP tools:
  - `dti_ratio(monthly_debt, monthly_income)`, where `monthly_debt` **includes PITI**;
  - `housing_ratio(monthly_payment, monthly_income)`;
  - `ltv_ratio(loan_amount, property_value)`;
  - `reserves(liquid_assets, monthly_payment, cash_to_close, required_months)`, which reports the shortfall when funds < cash to close;
  - `credit_score_tier(credit_score)`;
  - `large_deposits(deposits, monthly_income)`;
  - `total_obligations(debts, proposed_payment)`.
  - Each returns a short human-readable string starting with the number.

**`mcp_server.py`:**
- FastMCP app `underwriting-tools` (`mcp.server.fastmcp.FastMCP` in `mcp` 1.30.0) registers the 7 calculators plus `get_policy(section: str)` and `search_policy(query: str)`. Each policy result includes its section numbers.
- Run with `python -m underwriter.mcp_server` (stdio).

**`mcp_client.py`:**
- An async context manager opens a stdio client session, spawning the server with the current interpreter (`sys.executable -m underwriter.mcp_server`).
- `anthropic_tools()` converts the MCP tool list to Anthropic tool definitions: `name`, `description`, `input_schema`.
- `call(name, args) -> str`.
- **Simplicity choice:** open one session per specialist invocation. The spawn cost is acceptable, and it avoids lifecycle bugs (P1).

### 7.2 `llm.py`: Claude helpers (verify every call against the skill docs)

- **Client lifetime (P1).** There is no module-level client. `use_client()` is an async context manager that opens an `AsyncAnthropic(timeout=180, max_retries=1)`, puts it in a `ContextVar`, and closes it on exit. `graph.run_case()` and `graph.resume_case()` each wrap their work in it; the Critic service opens one for its own process at startup.
- **Request shape for every call:** `client.beta.messages.create` with the refusal fallback (hard rule 5), `thinking={"type": "adaptive"}`, an explicit `output_config.effort` (Opus 5.5 defaults to `medium`, so always set it), and `max_tokens=16000`. Opus 5.5 rejects disabled thinking and forced `tool_choice`; use `auto`.
- `async structured(system, user, schema: type[BaseModel], effort="high") -> (BaseModel, usage)`:
  - Structured output via `output_config.format`. Use `client.beta.messages.parse` if it exists in the installed SDK; otherwise `create` plus `schema.model_validate_json`.
  - Checks `stop_reason` (`refusal`, `max_tokens`) and raises a clear error.
- `async tool_loop(system, user, tools, call_tool, schema=None, max_rounds=8, effort="medium") -> ToolLoopResult(output, text, trace, status, usage)`:
  - A manual loop with `tool_choice` auto.
  - **Append the full `response.content` back each turn** (thinking blocks must be preserved).
  - Execute all `tool_use` blocks, in parallel when there are several, and return all `tool_result` blocks in one user message. A failed tool returns `is_error: true`.
  - When `schema` is given, the request also carries `output_config.format`, so the final `end_turn` message is schema-valid JSON and is parsed into `output`. No second call is needed.
  - If the cap is hit or the final JSON doesn't validate, `status="failed"` (fix P5).
- **Usage.** Both helpers return usage records `{agent, model, input_tokens, output_tokens}`. Nodes append them to the state key `usage`. `config.py` holds a small price table (Opus 5.5: $4 input / $20 output per million tokens). Unknown models show tokens only.

### 7.3 Agents and graph

**State** (`models.py`, a LangGraph `TypedDict`). Values are plain dicts (`model_dump()`), validated back into Pydantic models at node boundaries, so checkpoints always serialize.

| Key | Written by | Merge |
|---|---|---|
| `case_id`, `application` (sanitized only), `metrics` | `sanitize_and_measure` | replace |
| `analyses: dict[area, Analysis]` | specialists | dict-merge reducer |
| `tool_trace`, `bias_flags`, `usage`, `audit` | any node | append reducer |
| `supervisor_log: list[SupervisorStep]` | Supervisor | append reducer |
| `review: CriticReview \| None` | Critic sets it; Supervisor clears it when it dispatches a rework | replace |
| `rework_feedback: dict[area, str]` | Critic | replace |
| `rework_count: int` | Supervisor | replace |
| `decision`, `human`, `final_decision` | Decision / human review | replace |

**Rule: nodes that run in parallel (the specialists) write only reducer keys.** Every other key has exactly one writer per step. This is what makes the `Send` fan-out safe.

**Pydantic models:**
- `Analysis(area, status, failed_attempts, summary, findings: list[str], policy_refs: list[str], risk_level: LOW|MODERATE|HIGH, conditions: list[str], recommendation)`.
  - `status` is `complete` or `failed`. `failed_attempts` counts consecutive failures; at 2 the area is **exhausted** and isn't retried.
- `Defect(area, severity: major|minor, statement, should_be)`
- `CriticReview(defects: list[Defect], application_risks: list[str], policy_violations: list[str], summary)`
  - `verdict` and `rework_targets` are **not** model fields. `compute_verdict()` derives them: fail if any major defect; targets = the areas with major defects.
- `Decision(risk_score 0–100, decision, conditions: list[str], memo)`
  - The band is enforced in code: 0–30 APPROVED, 31–60 CONDITIONAL_APPROVAL, 61–100 DENIED. If the model's label disagrees with its score, use the band label and log it.
- `SupervisorStep(action: run_specialists|critic|decision, areas: list[str], reason, allowed: list[str])`

**Specialists** (`agents/specialists.py`):
- One `run_specialist(payload)`. The `Send` payload carries everything it needs: area, the relevant application slice, the authoritative tool inputs, and, on a rework, the feedback and the previous analysis.
- The prompt has:
  - the role and its policy section range;
  - the application slice;
  - the **authoritative tool inputs** from `metrics.tool_inputs()`;
  - an instruction to retrieve the policy sections it relies on and cite them;
  - when feedback exists, a **REWORK** block containing the Critic's defect(s) and the previous analysis. It must fix the defect and keep what was right (P3).
- One `tool_loop(..., schema=Analysis)` call produces the analysis.
- Results are screened with `bias.py`.

**Critic** (`agents/critic.py`):
- It judges **analysis quality only**: figures vs authoritative metrics, completeness, correct policy application, consistency across analyses, and fair-lending language.
- The prompt states plainly that disagreeing with a conclusion isn't a defect if the analysis applies policy correctly, and that borrower weaknesses go in `application_risks` / `policy_violations` (P4). It is told which areas are missing; a missing area is not a defect of the others.
- Exposed as an A2A service (`critic_service.py`):
  - The Agent Card has skill id `critic`.
  - Input is a JSON message `{case_id, application, metrics, analyses}`. Output is a JSON artifact with the `CriticReview` fields plus the computed `verdict`, `rework_targets` and the call's `usage` (so the main process can count the Critic's cost).
  - Failed tasks return a readable error.
  - Follow the `a2a-sdk` 1.1.2 API; it changed between versions.
- `a2a_client.py`:
  - `ensure_critic_running()` checks the Agent Card at `CRITIC_URL`; if absent, starts `python -m underwriter.critic_service` as a subprocess, waits for the card, and stops it at exit. If the port is held by something else, it raises a clear error. The app and the eval both use it.
  - `review(...)` sends the message and parses the artifact. Timeout **400 s**, which is longer than the Critic's worst case (180 s × 2 attempts).

**Supervisor** (`agents/supervisor.py`):
- `allowed_moves(state)` is pure code (the guardrails):

  | Situation | Allowed moves |
  |---|---|
  | No current review; some area is outstanding (never attempted, or failed and not exhausted) | `run_specialists` on any non-empty subset of the outstanding areas |
  | No current review; every area is complete or exhausted | `critic` if at least one analysis is complete, else `decision` |
  | Review passed | `decision` |
  | Review failed, `rework_count < MAX_REWORKS` | `run_specialists` on the review's `rework_targets` |
  | Review failed, `rework_count == MAX_REWORKS` | `decision` |

- `rework_count` counts rework rounds dispatched. `MAX_REWORKS = 3`, so the Critic runs at most 4 times.
- `choose_next(state)`:
  - One `structured()` call that sees a compact case summary, the allowed moves and the latest review, and returns a `SupervisorStep`.
  - If the pick isn't allowed, replace it with the first allowed move and log `guardrail override` (P2).
  - When it dispatches a rework, the Supervisor node clears `review` and increments `rework_count`.
  - Every step goes to `supervisor_log` with the `allowed` list. This log **is** the plan DeepEval grades.

**Graph** (`graph.py`), nodes:
- `sanitize_and_measure` → `supervisor`
- `supervisor` → conditional:
  - `run_specialists`: `Send("specialist", payload)` for each chosen area; parallel (P8). Each specialist → back to `supervisor`.
  - `critic`: `critic_node` (A2A client) writes `review` and `rework_feedback`, appends the Critic's usage → back to `supervisor`.
  - `decision`: `decision_node` → `human_review` if `needs_human_review()` (P7), else `END`.
- `human_review`:
  - `interrupt(payload)`, where the payload holds the decision, score, conditions, memo, review summary and the plain-language reasons for the review.
  - Resume value `{decision, notes, conditions, override_justification}`, checked by `validate_human_review()`.
  - Records `human`, `final_decision` and an audit line.
- **Entry points.** These are the only way the app, the eval and the tests run the graph:
  - `async run_case(case_id, thread_id, on_event) -> result`
  - `async resume_case(thread_id, review, on_event) -> result`
  - Each opens `llm.use_client()` and `AsyncSqliteSaver.from_conn_string("underwriter.db")` in `async with`, streams the graph, and calls `on_event(event)` for every step. Callers do `asyncio.run(run_case(...))`; a fresh event loop per call is safe because nothing outlives the call.
  - **Events** are plain JSON-serializable dicts with a `type`: `sanitized`, `supervisor`, `specialist`, `critic`, `decision`, `paused`, `final`, `error`. The UI renders them.
- `recursion_limit` 50.

**Decision** (`agents/decision.py`):
- `needs_human_review(decision, score, review, analyses) -> list[str]` implements the four P7 triggers and returns the reasons in plain words (empty list = no review).
- `validate_human_review(resume, ai_decision) -> str | None` returns an error message when the decision differs from the AI's and no justification is given. The UI calls it before resuming (to show the error in the form), and the node calls it again.
- The memo must include: score and decision; rationale; the key figures with policy section numbers; findings by area; the Critic summary; conditions mapped to the weakness each one fixes; and adverse-action reasons when DENIED.

### 7.4 `sanitize.py`, `bias.py`
- **Sanitizer:** mask the name to `[APPLICANT]`, SSN to `***-**-1234`, phone to `***-***-5678`, email, and street address. It runs **before** the graph, so the checkpointer never stores raw PII.
- **Bias screen:** a small term list of protected characteristics, with benign-context exceptions (e.g. "age of oldest tradeline"). It returns flags and doesn't block.

---

## 8. Streamlit app (`app.py`)

**Who it's for:** an employer or engineer who has seen the recording, cloned the repo and added an API key. They want to see how the agents cooperate and that a human stays in control. A live run takes a few minutes, so the page must show progress step by step, never a bare spinner.

### 8.1 Running a case
- The app has one mode: a live run. It needs `ANTHROPIC_API_KEY`. If the key is missing, the page shows one line on how to add it (and that a Claude subscription is not an API key), and the Run button is disabled.
- **On start:** `a2a_client.ensure_critic_running()`, once, via `st.cache_resource`.
- **Run:** the button handler calls `asyncio.run(run_case(case_id, thread_id, on_event))`. `on_event` appends each event to `st.session_state` and updates the timeline as it arrives.
- **One `render(events)` function** draws the page. A rerun (clicking a tab, typing in the form) redraws from `session_state` and never restarts a run.
- Keep `thread_id` in `st.session_state`. Resuming calls `asyncio.run(resume_case(...))`.

### 8.2 Layout
- **Sidebar:** case picker showing each case's one-line profile; the Run button; the model name; Critic service status (running / unreachable).
- **Top of the main area:**
  - key figures as `st.metric` tiles, each with its policy limit and section, and a status in words: "Back-end DTI 46.1% · limit 43% (2.5) · exception allowed". Never colour alone.
  - a five-stage progress line: Sanitize → Specialists → Critic → Decision → Human review, with the current stage marked.
- **Left column, the run timeline:** one expandable `st.status` entry per event, newest at the bottom:
  - Supervisor: the move picked, the reason, and the moves that were allowed;
  - Specialist: finished or failed, risk level, tools called with their arguments;
  - Critic: verdict; defects listed separately from borrower risks;
  - Rework: which analyst was sent back, and the exact feedback it was given.
- **Right column, result tabs:** Analyses (four sub-tabs) · Critic review · Decision memo (the full memo) · Privacy (before/after masking) · Audit and usage (audit trail; tokens and cost per agent, Critic included) · Evaluation.
- **Evaluation tab (read-only):** reads `eval/results.json` and shows the selected case's six DeepEval scores with pass/fail, and the deterministic checks. A caption gives the eval run's date and judge model and says: "These scores are from that evaluation run, not from the run on this page." If the file is missing or has no entry for the case, the tab shows the command to produce it. The app never runs DeepEval.

### 8.3 Human review
- A banner at the top: "Paused for human review", followed by **why**, in plain words with the policy section.
- The AI's recommendation, score and conditions sit beside the form, with the **decision memo shown in full** (not collapsed), because the reviewer needs it to decide.
- The form defaults to "Concur". Conditions are an editable list. The justification box appears only when the chosen decision differs from the AI's. Submitting an override without it shows the `validate_human_review()` message next to the box; nothing is lost and nothing resumes.
- After submit: a final panel with the AI recommendation, the human decision, an "Overridden" marker when they differ, the **decision memo in full**, and the audit lines.
- A case that ends without human review shows the same final panel, memo included.

### 8.4 When things go wrong
- Critic unreachable: the sidebar says so, and Run is disabled.
- A step fails mid-run: its timeline entry shows the readable reason; everything finished so far stays on screen; a failed specialist is shown as failed and then as escalated to human review.
- A run can't be cancelled from Streamlit. Say so next to the progress line; don't offer a button that doesn't work.

### 8.5 Constraints
One file of roughly 270 lines, stock Streamlit components, no custom CSS beyond `st.set_page_config(layout="wide")`.

---

## 9. Evaluation (`eval/run_eval.py`)

- Runs all four cases end to end through `run_case()`. Cases that pause are auto-resumed with the AI's own decision (concur), so the eval doesn't need a human.
- **Flag:** `--case CASE-00X` runs one case.
- Sets `DEEPEVAL_TELEMETRY_OPT_OUT=YES` and needs no Confident AI login.
- **Judge:** a `DeepEvalBaseLLM` subclass whose `generate` / `a_generate` call `llm.structured()` when DeepEval passes a `schema`, else return plain text. It uses `JUDGE_MODEL`.
- **Metrics** (DeepEval 4.2.2; check each constructor against that version's source before use):
  - TaskCompletion and StepEfficiency (from the full trace);
  - ToolCorrectness, with `evaluation_params=[ToolCallParams.INPUT_PARAMETERS]`, `available_tools` = all MCP tools, and `expected_tools` built **per specialist** from `metrics.tool_inputs()`. Exact-argument comparison covers the seven calculators. The two policy tools are checked for "called at least once", because their arguments legitimately vary.
  - ArgumentCorrectness (in 4.2.x it reads only `input` and `tools_called`);
  - PlanQuality and PlanAdherence, where the plan is `supervisor_log` and execution is the trace.
  - The four trace-based metrics may need DeepEval's own tracing instead of a hand-built trace; build step 0 settles how they are fed. If one can't be driven honestly from our trace, report it as unsupported; don't approximate it.
- **Deterministic checks**, printed alongside: decision ∈ acceptable set (section 6.2), human review triggered where expected, no bias flags, no reworks without major defects.
- **Cost:** before the first full run, extrapolate the cost from the measured CASE-001 run and show the owner.
- **Output:** a per-case table plus an overall PASS/FAIL, printed to the terminal, and `eval/results.json` written: the run date, the judge model, and per case the six metric scores with pass/fail and the deterministic checks. The file is **committed** after an approved run, and the app's Evaluation tab reads it (section 8.2). A `--case` run updates only that case's entry.

---

## 10. Tests (no API key, no network)

- `fake_llm.py` provides deterministic responses keyed by prompt role. The graph tests monkeypatch `llm.structured` / `llm.tool_loop` and the A2A client.
- **Required test coverage:**
  - metrics facts per case (section 6.2), including P6, and that results don't depend on today's date;
  - `tool_inputs` consistency with the calculators;
  - policy parsing and search; policy text contains every threshold constant;
  - sanitizer; the loader strips `expected`;
  - the MCP server via a real stdio session (local, no network);
  - `compute_verdict` (P4);
  - `allowed_moves` for every row of the table in 7.3, and guardrail override (P2);
  - decision band, all four `needs_human_review` triggers (P7), and `validate_human_review`;
  - graph routes: all pass; one rework that includes feedback in the prompt (P3); rework cap reached; a specialist that fails twice ends in human review, not a loop (P5); parallel specialists merge correctly and write only reducer keys (P8); Critic usage is merged into `usage`; pause and resume with an override requiring justification; persistence across a new `run_case` / `resume_case` call on the same SQLite file (P9);
  - the app (one `streamlit.testing.v1.AppTest` smoke test, with the fake LLM and a fake Critic client): the page loads; a case runs and pauses; an override without justification shows the inline error; with a justification, the final panel appears and contains the memo; the Evaluation tab shows scores from a results file and the how-to message when the file is missing.

---

## 11. README (rewrite the existing one)

**Sections:**
1. **What it does:** `demo.gif` at the top, then two paragraphs.
2. **Architecture:** a mermaid diagram. Supervisor ⇄ parallel specialists ⇄ MCP (stdio); Critic over A2A; Decision; human review gate; SQLite checkpoints; DeepEval.
3. **How a case flows.**
4. **Design choices:** the P1–P11 fixes, written positively ("specialists receive the Critic's defect on rework"). No criticism of the course.
5. **Run it:** `uv sync`, `cp .env.example .env`, add an Anthropic API key, then `uv run streamlit run app.py`. One line: a Claude subscription is not an API key.
6. **Tests and eval.**
7. **Sample results:** a table typed in from the real runs: case, outcome, human review, cost, and the six DeepEval scores.
8. **Limitations:** synthetic data and a synthetic policy; not lending advice.

Keep the old README's problem statement if it still fits.

---

## 12. Build order (stop at each checkpoint and report)

0. `.gitignore` update **first**. Then `pyproject.toml` with the pins from section 2 and `uv sync`. **Dependency check, no paid calls:** confirm by import and `inspect` at the pinned versions: `mcp.server.fastmcp.FastMCP`; the `a2a-sdk` 1.1.2 server and client entry points; `Send`, `interrupt` and `AsyncSqliteSaver` on langgraph 1.1.9; the six DeepEval 4.2.2 metric constructors and how the trace-based ones are fed; whether `client.beta.messages.parse` exists. **Checkpoint:** report the resolved versions and anything that differs from this brief.
1. `config.py`, `.env.example`, `data/policies.md`, `data/cases.json`, `metrics.py`, `policy.py`, `sanitize.py`, `bias.py`, and their tests. Run the local originality check (section 6.3). **Checkpoint 1a:** `uv run pytest` green; show the owner the policy, the case table and the originality result, and wait for their review.
2. `mcp_server.py` and `mcp_client.py`, plus the stdio test. **Checkpoint:** green.
3. `llm.py`, `models.py`, `agents/specialists.py`, `fake_llm.py`, and their tests. **Checkpoint:** green. Then ask the owner for approval to make **one live specialist call** on CASE-001, and report its usage.
4. `agents/critic.py`, `critic_service.py`, `a2a_client.py`, plus a test that runs the service in-process. **Checkpoint:** green.
5. `agents/supervisor.py`, `agents/decision.py`, `graph.py`, plus the graph tests. **Checkpoint:** green.
6. `app.py` and `test_app.py`. **Checkpoint:** green. Then, with approval, the owner or the agent runs all four cases live, and CASE-003 (or CASE-002) pauses and resumes.
7. `eval/run_eval.py` and the app's Evaluation tab. **Checkpoint:** green. Then, with approval, a live eval run; `eval/results.json` from that run is kept for commit.
8. README rewrite. The owner records a live run of one case with macOS screen capture, following a short shot list from the agent. The agent converts the file to `demo.gif` at the repo root with `ffmpeg` and references it in the README.
9. Report to the owner.

---

## 13. Done when

- `uv run pytest` passes on a clean clone with no `.env`.
- With a key, `uv run streamlit run app.py` underwrites all four cases. At least one pauses for review, and an override without justification is rejected.
- The token and cost summary includes the Critic.
- A specialist that fails twice ends in human review, not a loop.
- `uv run python eval/run_eval.py` prints six DeepEval metrics (or names any that this DeepEval version can't support), with argument values actually compared, plus the deterministic checks.
- The decision memo is shown in full in the Decision memo tab, beside the review form and in the final panel.
- The Evaluation tab shows the committed DeepEval results for each case.
- `git status` shows nothing from `docs/` or `backup/`, and no `.env` or `*.db`.
- The README shows `demo.gif` and a results table from real runs, and explains the system in under five minutes of reading.

---

## 14. Implementation notes (where the code differs from the text above)

- `llm.structured()` and `llm.tool_loop()` both use `client.beta.messages.create` with `output_config.format` and validate with Pydantic; `parse` is not used, so there is one request path. `llm.text()` was added for the judge's plain-text calls.
- Specialist prompts also carry an `<authoritative_facts>` block: the `case_metrics` values relevant to that area.
- `case_metrics` has extra keys: `qualifying_income_note`, `income_flags`, `loe_required`; `repair_flags` is a dict.
- The first graph node is named `measure`; sanitizing happens in `run_case()` before the graph starts.
- `graph.get_state(thread_id)` was added; `resume_case()` validates the review before resuming and raises `ValueError`, so an invalid review leaves the case paused.
- `SupervisorStep` also records `override: bool`.
- The eval results path is `config.EVAL_RESULTS`.
- **DeepEval trace metrics (step 0 question, settled):** in 4.2.2 TaskCompletion, StepEfficiency, PlanQuality and PlanAdherence serialize `LLMTestCase._trace_dict` to JSON for the judge. `eval/run_eval.py` sets it to the recorded run as nested spans (`name`, `type`, `input`, `output`, `children`), the shape `trace_manager.create_nested_spans_dict` produces. It is a private attribute, so recheck it if the DeepEval pin ever changes.
- ToolCorrectness is measured once per case over all specialists' calls, with the seven calculators as `expected_tools`. "Every analyst looked up the policy" is a deterministic check.
- A deterministic check was added beyond the four listed: every analyst called a policy tool.

