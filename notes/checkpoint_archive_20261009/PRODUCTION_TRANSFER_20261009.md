# B5 첫8개 체크포인트 실제 이관 — 8/8 완료

> **완료:** 23:09PDT 8/8 검증·이관·NVMe정리 완료, 오류0. 23:15PDT receipt/파일크기/원본부재/보호대상 유지 재확인. NVMe여유204.15GiB. [완료·C2우선 기록](C2_PRIORITY_20261009.md). 아래 실행중 기록은 이력이다.

사용자 지시: "이관 해". 목적지: 프롱혼. DeltaAI NVMe의 B5 `epoch_01`–`epoch_08` 약99.06GiB를 검증 후 이관한다. `latest_state`, 최신 두 완료 epoch13·14 및 best(epoch14)는 보호한다.

- 시작: 2026-10-09 20:15:20 PDT / 22:15:20 CDT (UTC2026-10-10T03:15:20).
- 실행 호스트: pronghorn `login-0`.
- runner PID: **20269**.
- 전용 tmux socket: **codex-archive-20261009**; session: **archive-b5-first8-20261009**.
- 실행 디렉터리: `/data/gpfs/assoc/pgl/data/Transgenic/ops/checkpoint_archive/transfers/b5-first8-20261009`.
- 상태: `status.json`; 항목별 검증·제거 결과: `transfer.jsonl`; stderr: `transfer.stderr`; wrapper 로그: `runner.log`.
- 코드: 검토·양쪽시험 완료 `3b4eb0dfe0e83463775ce8e1ab05fa2f62d1e20b`. 실행 직전 양쪽7파일SHA 재대조 PASS.
- scope: source `/work/nvme/bilv/wcyim/transgenic/runs/seed123`, destination `/data/gpfs/assoc/pgl/data/Transgenic/archives/checkpoint_archive`.
- 설정:16GiB/1checkpoint/pass, max_epoch8,16MiB/s, destination 최소1TiB 유지,8cycles/300초 간격. nice10/ionice best-effort7로 실행. 예상2–3시간(스토리지 상태에 따라 변동).
- 입력 config는 실행 디렉터리에 snapshot을 저장했으며 runner가 검증된 CLI의 `--apply --remove-source`를 호출한다. 사본의 SHA/내구성 확인 및 source 보호 재검사 뒤만 원본 정리.
- 완료 시 runner가 `status.json`을 `complete` 또는 `needs_attention`으로 갱신한다. CLI return0, epoch1–8 각 verified 및 removed 기록,8개 receipt가 모두 있어야 complete다.

## 재개/확인

**중복 제출하지 말고 status/프로세스/로그부터 확인한다.** 시작 검증에서 tmux 실행 중(TMUX0), runner state running, 첫 epoch01 계획·보호13/14,stderr빈값을 확인했다. 첫 기본 tmux 연결은 기존 서버/클라이언트 protocol version mismatch로 실행되지 않아, 프로세스·status가 없음을 확인한 뒤 위 전용 socket으로 실행했다. 기존 사용자 tmux 서버는 변경하지 않았다.

```bash
ssh pronghorn 'cat /data/gpfs/assoc/pgl/data/Transgenic/ops/checkpoint_archive/transfers/b5-first8-20261009/status.json'
ssh pronghorn 'tmux -L codex-archive-20261009 has-session -t archive-b5-first8-20261009'
```

완료 후 확인: `status.json`의8개 archive 경로/합계, 각receipt, DeltaAI epoch01–08 부재,epoch09–14/latest_state/best 유지, NVMequota 여유 증가. 아직 완료를 선언하지 않았다. Slurm 잡은 변경하지 않았다. 시작 직전 B5/WP5는 모두PENDING, B5link2 예상10-10 10:49CDT, WP5첫링크 예상10-10 22:25CDT였다.

## 20:17:55 PDT 진행 확인

runnerPID20269 alive, tmux session활성,오류0/stderr빈값. 첫체크포인트 stage에3파일·1,827,668,363bytes가수신됨(아직검증전). verified0/8,removed0/8. 실제복사작동확인. 이후상태는원격status.json/transfer.jsonl을확인한다.

## 22:23 PDT 확인

5/8(epoch01–05) 검증·정리완료, 약65.77GiB. receipt/파일크기/원본부재/보호대상 유지 재확인, 오류0, PID20269활성. NVMe여유170.86GiB. 나머지3개계속실행중. [학습·큐 상태](STATUS_20261009_2222PDT.md).
