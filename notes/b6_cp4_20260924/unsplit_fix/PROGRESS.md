# B6 unsplit compile 수정: 2026-09-24 20:43 PDT

## 최신: 2026-09-24 21:02 PDT, 정식 CP4 제출

- 새 메모리 벤치3208285 원본·로그 직접 검증 완료: PASS, 2ranks, 12steps/4warmup/8measured, 각rank error=null·compile_verified=true, peak reserved 약62.959 GB≤100 GB, optimize_ddp=false, Inductor/비재진입 checkpoint 유지. `bench3208285-verified.txt`.
- 같은 수정본f1ee718/원격9ad97de의 source217파일 재검증, 활성B6잡0, 잔액1023 확인.
- **정식 CP4 wp3-fixtures 잡3208442** 제출1회, GPU2/1h30m/ghx4-interactive. 21:02PDT PENDING(Priority), 예상시작N/A.
- 제출 저널 `CP4_UNSPLIT_WP3_SUBMISSION_ATTEMPT` 선기록, sbatch rc0. 원격repo `repo_b6_unsplit_20260924`.
- 로그 `/work/nvme/bilv/wcyim/transgenic/repo_b6_unsplit_20260924/b6-int4-3208442.out`.
- 보고서 `/work/nvme/bilv/wcyim/transgenic/runs/b6_int4/run_3208442/report.json`.
- 30분 확인 대상은3208442다. 보고서 status/failures/resume_byte_identical/generation·3link 결속과 run identity compile_policy의 optimize_ddp=False를 확인한다. PASS 뒤 synthetic 별도 제출. 현재synthetic 미제출.

## 진단 결과 직접 검증

진단3207863 COMPLETED 0:0, 6분7초. 두 case의 원본 JSON·로그·exit 및 전후 source217파일 검증을 직접 확인했다. `diagnostic3207863.txt` 참조.

- eager: report PASS, resume_byte_identical=true.
- unsplit Inductor(optimize_ddp=False): report PASS, resume_byte_identical=true. straight/resumed step3 파일 SHA256 모두 `05a396a95e5faac8f77ae9a41e86e16d19de6d38a2c0bd4d37969a14499270d0`.
- 원래 default partitioned Inductor CP4는 FAIL이었다. 통제된 비교에서 DDPOptimizer graph partition 설정을 끄면 재개 불일치가 해소됨을 확인했다. PyTorch 내부의 RNG/graph 순서 메커니즘까지 증명했다는 주장은 하지 않는다.
- 진단 자체는 정식 CP4 승인으로 사용하지 않는다.

## 최소 수정

로컬 `/tmp/b6-integration` 커밋 **f1ee7185caf706e59c144ee800cb7f3c6a7b1e83**, 4파일.

1. `src/transgenic/training/b6_runtime.py`: B6 `compile_b6`에서 `torch._dynamo.config.optimize_ddp=False`. DDP와 Inductor, suppress_errors=False는 유지한다. eager 전환 아님.
2. `train/train_HyenaTransgenic_B6.py`: 불변 `compile_policy` identity에 실제 optimize_ddp 값을 기록한다. 이전 compile policy의 체크포인트를 새 설정으로 몰래 재개하지 못한다.
3. `revision/scripts/b6/bench_memory_129k.py`: 결과 JSON에 optimize_ddp 값을 기록한다.
4. `tests/test_b6_trainer.py`: 실제 compiled forward/backward 및 이전 policy 재개 거부 회귀시험.

메모리·재개·생성 판정 기준, 데이터, 가중치 구조, 손실은 불변이다. 이전 작업의 생산 체크포인트를 사용하지 않는다. 컴파일 분할 변경은 성능/메모리에 영향을 줄 수 있으므로 **기존 메모리 벤치 PASS를 재사용하지 않고 새로 측정한다**.

## 검증

- 수정 전 신규 회귀2건 모두 FAIL(`b6-unsplit-red.log`).
- TF4.57.6 관련 스위트 **68 passed, 95 subtests passed**(`b6-unsplit-457.log`).
- TF4.46.3 trainer/bench 및 MVP 핀 포함 **37 passed, 78 subtests passed**(`b6-unsplit-446-pins.log`).
- 실제26.08 SIF CPU 신규 회귀 **2 passed**(`staging-sif.txt`).
- 독립 Codex agent `159ead6f-51b8-4b9`: **APPROVE for NEW MEMORY BENCH, then formal CP4 only after PASS**. 생산 승인 아님. 차단지적 없음.

## 새 벤치 제출

- 새 원격 repo `/work/nvme/bilv/wcyim/transgenic/repo_b6_unsplit_20260924`.
- 원격 스냅숏 **9ad97de801aa36f21452bc24a2663b2b0504b459**.
- 217개 source/config/test 파일 대조 통과. `source-f1ee718.sha256` 자체 SHA256 `9d01bc659a439cad041c5fd08eb20b088ccec14b99ebca9809294a84ed7ba890`.
- 제출 전 활성 B6잡0, 잔액1023 및 보수적 여유304 확인.
- **새 메모리 벤치 `3208285`**, GPU2/1h/ghx4-interactive. noclobber 저널 후 제출1회, sbatch rc0.
- 20:43 PDT PENDING(Priority), 예상 시작 N/A.
- 로그 `/work/nvme/bilv/wcyim/transgenic/repo_b6_unsplit_20260924/b6-mem129k-3208285.out`.
- JSON `/work/nvme/bilv/wcyim/transgenic/runs/b6_bench/mem129k-3208285.json`.

## 다음

30분 확인 대상은3208285다. rank별 실제 측정·compile 조건과 함께 JSON optimize_ddp=false를 확인한다. PASS 뒤 **새 repo에서 정식 CP4 wp3-fixtures**, 그 결과 PASS 뒤 synthetic. 현재 정식CP4/synthetic 미제출. 기존 실패 산출물·B5·운영 학습·main 병합·푸시 무변경.
