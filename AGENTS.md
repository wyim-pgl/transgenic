<!-- graft:start -->
## Graft — repo context graph

This repo is indexed in `graft/`: small linked markdown nodes that explain each
system and carry exact file:line spans, kept in sync with the code through git.

For ANY task here — understanding how something works, finding where code lives,
or scoping a change — get context from the graph before grepping or opening
source files. Re-ask freely (it's cheap) and reuse literal identifiers you
already have (symbol, error string, file name) as the query. New to this repo?
Run `graft map` first — a token-budgeted orientation (dir clusters, hubs,
hotspots), no LLM, no key.

- Run `graft ask "<your question>" --source` → ranked nodes with the relevant
  code spans inlined (each hit's ≤8-line crux by default; `--full` for whole
  definitions when the crux isn't enough). Match the tool to the task shape:
  for understanding or editing, the top node IS the answer — cite its
  `covers:` file:line spans and edit straight from `--source`. For
  exhaustive tasks ("every occurrence / every caller of this pattern"), ranked
  results are top-N, not complete — run `graft grep "<literal>"` instead
  (exhaustive over indexed files, grouped by enclosing symbol), falling back
  to raw `grep -rn` only for unindexed files.
- `graft skeleton <file>` → every definition's signature + span, ~10× cheaper
  than reading the file; use it to skim an API surface.
- `graft callers <symbol>` gives precomputed, exact edges — who calls this.
  Add `--direction out` for what it calls, or `--depth N` to walk
  transitively for the full blast radius. For structural questions, skip
  ranking and use this directly.
- Or browse: `graft/INDEX.md` lists every node; follow the links.
- Monorepos and folders of multiple repos rank fairly across sub-projects —
  hits carry `[scope/]` labels naming which one they're from. Narrow with
  `graft ask "<task>" --in <scope>/` once you know where you're working.

If a returned span is truncated ("+N more lines"), open the file at that exact
range before finalizing. Only open source files when a node genuinely lacks a
needed detail, and then at the exact file:line the node points to — never
re-read whole files.

After big code changes, refresh the graph with `graft build` (deterministic,
no API key, $0).
<!-- graft:end -->

## Session recovery

When resuming interrupted work, read `.atl/resume.md` first, then the current top checklist in `.atl/todo.md`; consult `.atl/quarantine.md` before repeating a failed approach. Follow the latest handoff linked from resume for current branches, agents, and job IDs; older handoffs are historical snapshots. Verify live Slurm status before resubmitting jobs. Implementation branches are not merged into main; build/query graft in the selected worktree to see those changes.

## Continuous TODO execution

현재 `.atl/todo.md`에 미완료 작업이 있으면, 기존 작업 범위 안에서 사용자에게 추가 승인이나 “계속할까요?”를 요청하지 않고 다음 실행 가능한 항목을 계속 진행한다. 서브에이전트 완료 알림 뒤에도 결과 확인 → 필요한 수정 → 검증·독립검토 → 후속 작업으로 연결하며, 진행 보고만 하고 작업을 멈추지 않는다.

명시적인 승인 대기, 자원 예산, 안전·과학적 검증 조건은 자동 승인하거나 우회하지 않는다. 막힌 항목은 이유와 필요한 결정을 기록하고 다른 실행 가능한 TODO를 진행한다. 실행 가능한 항목이 모두 완료되거나 차단되면 상태를 보고한다. 완료·차단·다음 단계와 실제 커밋/잡 ID는 `.atl/todo.md` 및 `.atl/resume.md`에서 연결된 진행 기록에 갱신한다.
