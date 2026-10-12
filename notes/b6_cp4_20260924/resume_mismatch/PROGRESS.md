# CP4 재개 불일치 조사: 2026-09-24 19:47 PDT

## 확인된 사실

원래 CP4 `3207331`은 FAILED 1:0, 실행6분4초. 보고서의 유일한 최종 실패는 `resume determinism: step checkpoints differ`다. 세 링크의 입력 결속 기록은 같고 생성 시퀀스 parity는 통과했지만 CP4 전체는 FAIL이다. 원본 `failure3207331.txt` 보존.

큰 텐서는 원격 CPU1스레드에서 safetensors 및 torch mmap으로 읽기 전용 비교했다. `artifact-summary.json`과 `artifact-comparison.txt` 참조.

- epoch01 모델: **770/770 tensor bitwise equal**.
- step00003 모델: **622/770 tensor different**, 최대 절대 차이 `8.833850733935833e-05`.
- epoch01 아직 적용하지 않은 기울기: **각 rank 610개 tensor different**. rank0 evidence projection 최대 차이 `0.008026123046875`.
- epoch01 저장 Python/NumPy/torchCPU/CUDA RNG 상태: 각 rank 내부에서 straight/resumed **동일**.
- 각 rank의 전체 draw log 바이트·해시도 **동일**(`order-check.txt`).

따라서 단순 파일 직렬화 차이가 아니다. 가중치가 아직 같을 때 epoch-tail backward가 달라진 사실이 확인됐다. RNG 소비 횟수 차이만으로 원인을 단정하지 않는다. 같은 counter를 다른 compiled stochastic op에 배정하는 현상, DDP graph partition 등은 아직 가설이다. 데이터 내용 전체의 바이트 추적을 했다는 주장도 하지 않는다.

## CPU 재현

기존 real trainer와 tiny model fixture로 accumulation tail을 포함한 straight/preempt/resume을 비교했다.

- 로컬 torch2.6 CPU Inductor는 모델 codegen에서 `AssertionError: buf361`로 실패. 운영 원인 증거로 사용하지 않는다.
- 실제26.08 SIF CPU는 최초 CXX 환경변수 `CC` 때문에 compiler 탐색 실패. SIF에 존재하는 `/usr/bin/g++`를 명시해 재실행했다. GPU 배포 설정을 바꾼 것이 아니다.
- 실제 SIF CPU Inductor 재현은 **epoch01 weights, pending gradients, step3 weights 모두 바이트 일치**. 즉 GPU/DDP 실패를 CPU1rank로 재현하지 못했다. `sif_cpu_inductor_gcc.txt`.
- 두 읽기 전용 Codex 조사도 compiled stochastic graph / DDP 분할 등을 후보로 제시했으나 원인 확정은 아니다. Advisor 호출은429 rate limit으로 실패.

## 제한된 GPU 진단

같은 실패를 무조건 재실행하거나 합격 기준을 낮추지 않고 두 조건을 한 잡에서 비교한다.

1. **eager:** backend none, 나머지 real model/wp3-fixtures/2ranks/bf16/비재진입 checkpoint/결정성 설정 동일.
2. **unsplit:** backend inductor, `torch._dynamo.config.optimize_ddp=False`만 추가.

기존 실패는 baseline Inductor+기본 DDP optimizer다. 각 조건은 별도 accelerate 프로세스 및 별도 출력 디렉터리를 사용한다. 모델 데이터/학습값 변경 없음. 이 실행은 **진단 전용**이며, case report가 PASS여도 운영 CP4 통과 또는 synthetic 착수 허가로 사용하지 않는다.

- 스크립트 `run_probe.py`, `probe.sbatch`.
- 독립검토 초기 차단(원본 소스/cache 쓰기 가능)을 수정: repo readonly mount, bytecode 쓰기 금지, per-job209MB HF cache 사본, 각 case 후 소스 해시 재확인.
- 재검토 agent `8972756d-3931-4d6`: **APPROVE**, 구체적 차단 사항 없음.
- 원격 source는 `repo_b6_devicefix_20260924`, 로컬 `a48579d` / 원격 `6446692`. 생산 코드 수정·추가 커밋 없음.
- **진단 잡 `3207863`**, GPU2, ghx4-interactive, 상한30분. 사전 B6 활성잡0, 잔액1024, source217파일 해시 확인. 제출시도 저널 선기록, sbatch rc0, 제출1회.
- 19:47 PDT 실측 **PENDING(Priority)**, 예상 시작 N/A.

### 확인할 경로

배치 로그:
`/work/nvme/bilv/wcyim/transgenic/diagnostics/resume_mismatch_20260924/b6-resume-diag-3207863.out`

결과 root:
`/work/nvme/bilv/wcyim/transgenic/runs/b6_resume_diagnostic/job_3207863/`

그 아래 `case_exit_codes.txt`, `eager.log`, `unsplit.log`, `eager/report.json`, `unsplit/report.json`, `source-after-eager.txt`, `source-after-unsplit.txt`를 함께 본다. 배치 최종 rc0은 두 case 성공을 의미하지 않는다. 실패도 수집하기 위해 각 case rc를 따로 기록한다. 일부 case가 끝나지 않으면 미측정으로 보고한다.

## 다음 단계

30분 확인 대상은 진단 `3207863`이다. 결과에 따라 가설을 검증하고 최소 수정·회귀 검증·독립검토 뒤 정식 CP4를 다시 실행한다. 합격 기준(재개 후 checkpoint byte identity)을 유지한다. 현재 CP4와 synthetic은 차단 상태다. B5·운영 학습·기존 산출물·main 병합·푸시 무변경.
