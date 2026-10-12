# CP4 시작: 2026-09-24 17:32 PDT

## 선행 메모리 벤치 검증

`3207080`의 원본 JSON과 로그를 본 세션에서 직접 읽고 검증했다. 상태 PASS, world_size=2, rank 집합 {0,1}, steps=12/warmup=4, 양 rank measured_updates=8, compile_verified=true, error=null, peak_reserved=33,153,875,968 bytes. GPU당 33.154 GB로 한도 100 GB 이하다. sacct COMPLETED 0:0, 3분15초.

재컴파일 계수 자체는 JSON에 없다. 대신 실제 실행 소스 SHA256을 로컬과 대조하고 다음 경로를 확인했다.

- `revision/scripts/b6/bench_memory_129k.py:165-166`: 매 optimizer step마다 CompileGuard.observe 호출.
- `src/transgenic/training/b6_runtime.py:462-479`: warmup 이후 같은 shape에서 graph 증가 또는 suppress_errors=true이면 RuntimeError.
- 벤치의 예외 처리와 게이트: guard 예외는 rank.error에 기록되고 PASS 불가. 양 rank는 error=null, 측정8회 완료.

따라서 **현재 구현된 CompileGuard 기준의 warmup 이후 graph-count 증가 거부 조건을 통과했다**. 모든 가능한 컴파일 이벤트의 독립 계수가 0이라는 더 강한 주장은 하지 않는다. 새 조건 추가나 기존 게이트 완화 없이 CP4 선행조건을 충족했다. Advisor 도구 실패는 기록된 검토 불가 상태이며, Advisor의 성공 응답을 받았다고 주장하지 않는다.

원본 증거: `bench3207080-evidence.txt`.

## WP3 fixtures 통합 시험

- **잡 `3207149`**, `DATA=wp3-fixtures`, GPU 2, ghx4-interactive, 상한 1시간30분.
- 17:32 PDT 실측 PENDING(Priority), 예상 시작 N/A.
- 코드: 로컬 `c9e242c`, 원격 `7018e33`. 216개 소스·설정·시험 해시 재검증 통과.
- 경로 `/work/nvme/bilv/wcyim/transgenic/repo_b6_cachefix_20260924`.
- 로그: 위 경로의 `b6-int4-3207149.out`.
- 최종 보고서: `/work/nvme/bilv/wcyim/transgenic/runs/b6_int4/run_3207149/report.json`.
- 제출 전 활성 B6 잡 없음, 잔액1,024 확인. noclobber 제출시도 저널 `CP4_WP3_SUBMISSION_ATTEMPT`, 결과 `CP4_WP3_SUBMISSION_RESULT`. sbatch rc0, 1회만 제출.

## 다음

예약 확인 대상은 이제 **CP4 `3207149`**다. 메모리 벤치 JSON 형식을 CP4 report에 적용하지 않는다. CP4 driver의 보고서/로그와 실제 종료 상태로 compile, checkpoint resume, cached generation parity를 확인한다. 통과 후 같은 수정본의 `DATA=synthetic` CP4를 별도 제출한다. synthetic은 아직 제출하지 않았다.

B5·운영 학습·원본 실패 디렉터리·main 병합·푸시는 무변경이다.
