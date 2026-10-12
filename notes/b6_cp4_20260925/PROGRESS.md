# 정식 WP3 CP4 검증 및 synthetic 착수: 2026-09-25 00:04 PDT

## 최종: 2026-09-25 01:03 PDT

**메모리3208285, WP3 CP43210279, synthetic CP43210585 원본 검증 모두PASS.** [최종 검증 정본](FINAL_VERIFICATION.md). 활성B6잡 없음, 새 제출 없음. 운영학습 승인/실행 아님. 아래 착수 기록은 이전 시점이다.

## WP3 fixtures CP4: 확인 완료

잡 **3210279 COMPLETED 0:0**, 6분31초. 본 세션이 원본 JSON·로그·run_config·checkpoint bytes를 확인했다.

- 실제 2-GPU/NCCL 사전 fault 시험 **7 passed**(65.34초).
- 정식 CP4 report PASS, failures=[], resume_byte_identical=true.
- straight/resumed step3 SHA256 모두 `05a396a95e5faac8f77ae9a41e86e16d19de6d38a2c0bd4d37969a14499270d0`.
- generation rank집합{0,1}, greedy/beam2 시퀀스 일치. 세 link 입력 결속 해시 동일.
- 두 run_config 모두 world_size2, bf16, nonreentrant, compile_policy=Inductor/suppress_errors=False/optimize_ddp=False.
- 메모리 벤치3208285와 모델·bench·config 바이트 일치 및 compile_b6 AST 일치 확인, bench config_sha도 실제 config에 결속. 운영학습용 config gate_status를 수정하거나 생산 승인을 새로 부여한 것은 아니다.

원본 증거 `wp3_verified.txt`, `gate_linkage.txt`.

## Synthetic CP4

선행 검증 뒤 같은 소스의 DATA=synthetic을 제출했다.

- **잡3210585**, GPU2/1h30m/ghx4-interactive.
- 00:04PDT 실측 PENDING(Priority), 예상시작N/A.
- 소스 `/work/hdd/bilv/wcyim/transgenic/repo_b6_filestore_20260924`, 로컬3511a5a/원격e2a91b7.
- source220파일·wrapper 해시 재검증, 활성B6잡0, 잔액1022, HDD quota/물리여유100GiB이상 확인. noclobber 저널 후 제출1회, sbatch rc0.
- 동일 wrapper로 NCCL 사전시험7사례를 통과한 뒤 synthetic 정식 통합시험을 실행한다.
- synthetic 런처 조건: tier30720, rows48, accumulation8(global), n_pilot5, stop_at3. compile/checkpoint/bf16 조건은 유지된다.
- 로그 `/work/hdd/bilv/wcyim/transgenic/repo_b6_filestore_20260924/b6-int4-3210585.out`.
- 보고서 `/work/hdd/bilv/wcyim/transgenic/runs/b6_int4/run_3210585/report.json`.

## 다음

30분 예약은3210585를 확인한다. NCCL7PASS 후 report status/failures/resume_byte_identical/generation, 각 run_config의 compile_policy와 재개 step5 hash를 검증한다. synthetic에는 WP3 전용 link_record_sha256 필드가 없으므로 이를 요구해 정상 결과를 잘못 거부하지 않는다. 결과에 없는 검증은 수행했다고 주장하지 않는다.

B5·운영 학습·기존 산출물·main 병합·푸시 무변경. NVMe147GiB 정리는 여전히 별도 승인 대기다.
