> 📌 **이 문서는 09-21 인증복구 시점의 기록이다. 현재 정본은
> [../a46_approval/PROGRESS.md](../a46_approval/PROGRESS.md)** — 그 뒤 FLNC/ONT 정책 4건이
> 저자결정으로 종결되고 A46 승인준비로 국면이 바뀌었다.

# 진행 정본 — 2026-09-21 인증복구 후 실측/4트랙 착수

모든 시각 PDT. 이 문서가 2026-09-21 현재 정본이며, `notes/auth_resume_20260917/PROGRESS.md`의 차단 상태를 아래 실측으로 갱신한다. 갱신되지 않은 항목은 09-17 기록이 그대로 유효하다.

## 1. 인증 복구와 접속 경로

사용자가 NCSA Kerberos + Duo 인증을 완료했다. `dtai`는 로컬 ssh config에 없고 pronghorn의 `~/.ssh/config`에만 존재한다(`HostName dtai-login.delta.ncsa.illinois.edu`). 따라서 접속은 2단 홉이며, 모든 원격 호출은 timeout으로 경계를 둔다.

```
timeout 180 ssh -o BatchMode=yes -o ConnectTimeout=15 pronghorn \
  'timeout 140 ssh -o BatchMode=yes -o ConnectTimeout=25 dtai "<remote command>"'
```

인증 후에는 BatchMode 홉도 통과한다. 상시 monitor나 반복 재시도는 두지 않는다.

## 2. B5 seed123 체인 — 2링크 추가 완료 (신규 사실)

09-17 이후 처음 확인한 실측. `sacct -u $USER -S 2026-09-17 -X`:

| JobID | JobName | State | ExitCode | Start → End | Elapsed | Node |
|---|---|---|---|---|---|---|
| 3131538 | b5-seed | COMPLETED | 0:0 | 2026-09-17T01:37:29 → 2026-09-17T12:24:21 | 10:46:52 | gh024 |
| 3149356 | b5-seed | COMPLETED | 0:0 | 2026-09-18T20:06:07 → 2026-09-19T06:54:56 | 10:48:49 | gh134 |
| 3149357 | b5-seed | COMPLETED | 0:0 | 2026-09-19T07:33:51 → 2026-09-19T18:20:47 | 10:46:56 | gh102 |
| 3149358 | b5-seed | PENDING (Priority) | 0:0 | — | — | — |
| 3163986 | mvp8-final-p1-0916 | FAILED | 1:0 | 2026-09-17T18:47:46 → 2026-09-17T18:48:37 | 00:00:51 | gh077 |

`squeue` 현재: 3149358 Priority 대기, 3149359–3149363 Dependency 대기. 남은 링크 6개.

- seed123 학습은 **전체 완료가 아니다.** 체인 진행 중일 뿐이다.
- 3163986은 terminal FAILED 그대로다. **취소/재명명/재제출 금지.**
- 이번 확인에서 기존 잡을 변경하거나 새 잡을 제출하지 않았다.

## 3. GPU staging I/O 차단 — 해소, 단 release는 INCOMPLETE (신규 사실)

09-17 17:54의 `gpu/STAGING_IO_BLOCKER.md`가 요구한 **안전조건 2개가 충족**됐다.

- **PID 1944304 종료 확인** — `ps -p 1944304` → `PID_1944304_GONE`. `pgrep -u $USER -a python3` 결과 없음(다른 스테이저 없음).
- **원격 파일 I/O 복귀** — staging 트리 `ls -la`/`stat` 즉시 응답. 09-17의 45s/160s 정지 재현되지 않음.

staging 경로: `/work/nvme/bilv/wcyim/transgenic/mvp_est8_20260916_1545/staging/dev123_p1_release_v6_20260917/`

| 파일 | 크기 | 모드 | 상태 |
|---|---|---|---|
| admission.py | 27611 | 0444 | 존재 |
| authorization.example.json | 2608 | 0444 | 존재 |
| build_release.py | 5688 | 0444 | 존재 (09-17 정지 대상 fd4) |
| code.candidate.tar | 1873920 | 0444 | 존재 |
| controller_execution_scope.json | 3332 | 0444 | 존재 |
| deploy.py | 29149 | 0444 | 존재 |
| notes/ | dir | — | 존재 |
| operator-pins.json | 11828 | 0444 | 존재 |
| p1.sbatch | 2010 | 0444 | 존재 |
| release.json | 25219 | 0444 | 존재 |
| **release.sha256** | **0** | 0660 | **INCOMPLETE** (mtime 2026-09-17 19:46:20 -0500) |
| **source-manifest.json** | — | — | **부재** |

**판정:** 차단은 해소됐으나 릴리스는 여전히 불완전하다. 다음 단계는 blocker 문서가 규정한 **명시적 기록 정합화**(불변 아카이브·핀된 ledger 대비 전 파일 해시 검증 + 미완성 2파일의 기록된 reconciliation + 완료된 연산 회계)이며, blind 재복사·재빌드·삭제·대체저장소 우회는 금지다. **GPU 제출은 정합화와 live gate 통과 이전 금지.** 이번 확인에서 원격 쓰기는 하지 않았다.

## 4. 미해소 유지 항목

- **ONT full** — 6165965 기술 결과만 수용. `.extern OOM 0:125` 미해결, 관리자 로그 필요, support draft 미발송. CLEAN/F4/full 차단 유지. 기술 성공이 운영 게이트를 면제하지 않는다.
- **panel transport** — 167파일 준비/8시험 PASS이나 유효한 안정 백업과 독립 transport 검토 대기.
- **EST/A11** — 16 GenBank 레코드 검증·서열 일치 확인 완료. 추가 BioSample 3건은 SSL timeout(curl 28)으로 **미확인**이며, 미확인을 부재로 판정하지 않는다. A11 전체 audit/exclusion/alias 정합화, routeA 실 trace fixture, 추가 producer 참조 SHA 연결 잔여.
- 과학적/A11-complete/production-raster 수용은 여전히 **false**.

## 5. 이번 착수 — 사용자 선택 4트랙 병행

사용자가 4개를 모두 선택해 독립 담당으로 동시 착수했다. 각 담당은 전용 폴더에만 기록하고, 잡 제출·커밋·병합·푸시는 하지 않는다.

1. **GPU staging 정합화** → `gpu/STAGING_RECONCILIATION.md`
2. **eval decoder repin** → `eval_decoder/REPIN.md`
3. **ONT full wrapper·실 fixture** → `ont/WRAPPER_AND_FIXTURE.md`
4. **A11 audit 정합화 + junction-counts 결속 + routeA 실 trace + spec 계약/native crash 증거** → `a11/AUDIT_AND_JUNCTION.md`

공통 금지: Slurm 제출, 기존 잡 변경, 체크 약화, staging 재빌드/삭제/대체경로, main 병합·푸시, 부분 수용의 전체 완료 확대.

## 6. 로컬 상태

main `9a8c8e1`, 미커밋 112항목 보존. worktree 11개 존속 — `est-approved-contract-20260917`/`est-real-source-binding-20260917`(fca911e), `mvp-seed123-integration-20260917`(9603e20), b6-p2 계열 3개, pi-agent 계열 4개. 구현 브랜치 미병합/미푸시 유지.
