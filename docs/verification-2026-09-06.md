# Verification — 6 September 2026

## Current-season sources

`python src/source_check.py` passed using fresh network downloads and the production current-season normalization path:

| Input | Observed |
| --- | --- |
| Season | 2026-2027 |
| Current Gameweek | 3 |
| Finalized Gameweeks | 1, 2 |
| Teams | 20 |
| Players | 653 |
| Fixtures | 380 |
| Finalized player-history rows | 1,268 |
| Official live player overlay | Available, 653 players |
| Official live fixture overlay | Available, 380 fixtures |

Primary source: [FPL-Core-Insights season summary](https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data/2026-2027/gameweek_summaries.csv). Official overlays: [players and Gameweeks](https://fantasy.premierleague.com/api/bootstrap-static/) and [fixtures](https://fantasy.premierleague.com/api/fixtures/).

This verifies current-season ingestion. It does not establish future availability or independently validate every historical or cup archive. The source-health check uses temporary storage and does not replace tracked data.

## Local validation

- Python 3.11.16; dependencies installed from requirements.txt.
- 49 tests passed, including cross-caller finalization parity, stdlib-only workflow gating, chip revision with transfer carry, model repair/failure handling, and successful pipeline skips.
- `python scripts/check_pipeline.py` passed with the real saved models, projection engine and CBC optimizer; produced 15 squad players, 11 starters, one captain and a UTF-8 lineup in isolated temporary storage.
- Compilation and `git diff --check` passed.
- Actionlint 1.7.12 passed for all workflows. ShellCheck was unavailable and disabled for this local invocation.

The pipeline smoke check supplies cached inputs and simulates an actionable deadline; it does not contact sources or publish outputs. Network normalization is verified separately above.

## Existing GitHub runs inspected

| Workflow | Evidence | Result |
| --- | --- | --- |
| CI | [20 August run](https://github.com/truong-tt/fpl-ml-manager/actions/runs/32396119524) | Dependency installation, compilation and tests executed successfully |
| FPL Daily Update | [5 September run](https://github.com/truong-tt/fpl-ml-manager/actions/runs/33986361757) | Pipeline, summary publication and commit steps executed successfully |
| Season Replay | [1 September run](https://github.com/truong-tt/fpl-ml-manager/actions/runs/33553282587) | Actual replay step ran 20:06:09–20:07:10 UTC and succeeded |
| Season Replay | [5 September run](https://github.com/truong-tt/fpl-ml-manager/actions/runs/33986444492) | Successful gate-only run; replay was skipped because no new Gameweek had finalized |

Evidence comes from public GitHub job-step metadata. Raw log downloads returned HTTP 403. The official checkout and setup-python v7 tags were verified to exist.

These runs predate this refactor. During the initial local review, GitHub authentication was unavailable. They establish historical workflow execution only; hosted validation of the revised workflows must be checked against the pushed commit.

## Workflow changes

- CI now supports manual dispatch and exercises the isolated model-to-lineup path.
- Source Health checks current-season normalization daily and on manual dispatch.
- Daily lineup publication requires an explicit generation result from that invocation.
- Data staging no longer suppresses all errors; lineup files are staged only after generation.
- Replay skips are visible in the job summary.
