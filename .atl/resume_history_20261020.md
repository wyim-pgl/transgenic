# Resume — 2026-10-20 C2 파일럿 완료 확인

**최신 정본: [RESTART_HANDOFF_20261012](../notes/RESTART_HANDOFF_20261012.md).** 이 파일 → [todo](todo.md) → handoff → [quarantine](quarantine.md) 순서로 읽는다. 과거 handoff는 현재 상태가 아니다.

## 사용자 지시

- C2 우선으로 원인을 수정하고 검증한다.
- **문제 해결·검증 통과 후 추가 승인 없이 C2 본학습을 바로 시작한다(10-12 명시 승인).** 과학적 게이트는 유지한다.
- 전체 작업 푸시 및 이 네 문서 갱신 후 clear 가능한 상태로 보존한다.

현재 정본: [컴파일 이력 복구 실행 기록](../notes/C2_REPLAY_PROGRESS_20261012.md). **4GPU 동일성 검증 3369694 PASS → 400/410 검증 3370048 PASS → 본학습 3379630, 4099/4099 완료.** 운영 정본 `ops/c2_in_place_198ce14_20261012`는 `complete` / revision22 / error 없음. 감시 프로세스는 완료 후 종료했으므로 재시작하지 않는다.

## 현재 상태

**10-20 10:44–10:46 PDT 실측:** SSH 정상. C2 검증·본학습 모두 parent/batch/extern `COMPLETED 0:0`. 현재 실행 중인 학습 잡은 없고 B6 두 잡만 held 상태다.

- **3364480 COMPLETED0:0**, 4GPU/gh150,39m17s. 동일 모델/버퍼/입력/RNG에서 컴파일 이력 재현→기존 손실과 일치; Dynamo만reset→cold logits와일치.12report독립검토PASS. 원인 확인 완료. **컴파일복구198ce14/로컬검증/독립리뷰 완료, NGC58시험57pass/1skip, 4GPU3369694 전체상태동일성PASS**.
- 앞선3361666: old/new 코드200→210 완전일치; 연속/재개210→220은609/770모델텐서불일치. 입력·난수복원실패는 아님.
- 검증 **3370048**: 10-14 07:28:53 PDT 종료, 2h01m02s / gh114. 400 저장 후 새 프로세스 400→410과 연속 학습의 모델·optimizer·scheduler·4rank RNG·모든 trace·history 정확 일치. rank3 slot6718 관측 및 역사적210 loss 복원 PASS.
- 본학습 **3379630**: 10-15 09:56:07 → 10-16 06:18:31 PDT, 20h22m24s / gh114. 최종 `step_04099`, `PILOT_FINAL`, `PRUNE_EXEMPT`, `TRAINING_DONE`, epoch 평가 및 allocation postflight 완료. 최신 복구용 `latest_state`는4000이며 최종4099 결과는 별도 디렉터리다.
- 실제 결과: `/work/nvme/bilv/wcyim/transgenic/runs/c2_compile_history_validation/3370048/resumed/step_04099`. 원래 C2 경로는 checkpoint200, 진단 사본은256 그대로 보존. 구C2후속3345920은 검증 후 취소됨.
- **B5 3309786 FAILED 1:0**, 10-15 04:49:05 PDT, 3m09s. 학습 전 검사에서 C2 호스트 시험용 절대경로 symlink가 컨테이너 안에서 dangling으로 판정돼 중단. 후속3309787–90 취소. 10-20 원인 재현 후 해당 시험용 링크만 제거했고 실제 컨테이너에서 제거·대상 receipt 보존 확인. B5 전체 preflight·재개는 아직 미검증/미제출.
- **B6 3345917/18 JobHeldUser, Nice0**. controller가 우선순위 복구를 마쳤지만 기존233574a 소스/런처/입력 검토와 교체 계획은 별도 과학적 게이트로 남는다. 자동 hold 해제 금지.
- 수집 증거35개와 [부모 검수 기록](../notes/checkpoint_archive_20261009/c2_failure_3345919/resume_determinism/status_20261020/PARENT_ACCEPTANCE.json), [B5 시험 링크 정리 기록](../notes/checkpoint_archive_20261009/c2_failure_3345919/resume_determinism/status_20261020/B5_FIXTURE_CLEANUP.json). 무거운 모델/입력 해시는 allocation postflight 증거를 사용했으며 login에서 다시 계산하지 않았다.

## 코드·다음 행동

**10-20 B6 후속 작업 재개:** 사용자가 B6 미완료 이유를 물어 후속 누락을 확인했다. 현재3345917/18은 실행0/held이고3345916은 입력 검증 잡이다. 원래 실패3306594와 이후 B6 smoke3338555(247업데이트)는 별도다.198ce14의 B6 CPU/aot_eager 재개 시험8개 PASS(141.281s), 실제 B6 evidence-on GPU 동일성은 아직 미증명. [B6 실행 계획](../notes/b6_pilot_resume_20261020/PLAN.md)을 진행 중이다. 구현 worktree `/tmp/b6-pilot-proof-20261020`, branch `b6-pilot-proof-20261020`, base19049ba; agent `/root/b6_proof_implementation`이 새 증명 도구를 담당한다. 기존source198ce14와 C2완료물은 불변. NVMe여유94.37GiB로100GiB 게이트 미달; B5epoch09–11(39.46GiB)을 검증 후 HDD에 이관하는 별도 CPU allocation 설정을 준비·검토 중이며 아직 새 잡은 제출하지 않았다. 이관은12–14/latest/best를 보존한다. 실제 후속 잡ID/완료 여부는 계획의 진행 기록과 live상태로 확인한다.

**10-20 16:36PDT 갱신:** 위 준비 상태에서 진행됨. 이관 **3425279 RUNNING/gh027**,16:35:01PDT시작,1GPU/1CPU/2GiB/2h. DeltaAI가zeroGPU제출을거부하므로최소1GPU할당으로CPU파일I/O만수행한다. 새operation`ops/b6_headroom_20261020_v2`, manifestc61be6d6…;v1은test-only거절/미제출보존. 중복제출금지. B6증명도구 **19d319d**(구현48d526b)완료,17proof+5hostcontract+38source시험PASS; `/root/b6_proof_review`가독립검토중. 아직B6 GPU증명/본파일럿새잡은없다. 계정잔액1641Hours확인. 다음은이관완료실측→증명검토/NGC검증→새B6증명잡→검증후자동4099연결도구구현·검토(Task4brief작성됨).

**Delta CPU 자원 신청:** 사용자가 NCSA Delta CPU **1,000core-hours**를 신청했고 처리 후 알리기로 했다(1,000ACCESScredits, 아직 활성화 확인 안 됨). DeltaAI GPU잔액1,641hours는별도다. DeltaAI는zeroGPU배치잡불가; 별도Delta의cpu파티션은가능하고/work/hdd·/work/nvme는공유된다. 승인·CPU Slurm계정·기존bilv경로접근을확인한뒤CPU전용이관/해시등을Delta CPU로전환한다. 신청을활성자원으로간주하거나현재3425279를자동취소하지않는다. B6실제4GPU검증/학습은DeltaAI에유지한다.

사용자 요청으로 [Delta CPU 사용 안내](../notes/delta_cpu_20261020/USAGE.md)와 [읽기 전용 사전 확인](../notes/delta_cpu_20261020/delta_cpu_preflight.sbatch)을 준비했다. cpu/1core/1GiB/10min, 실제계정은accounts로확인하여CLI지정. Bash·Python3.6구문/Slurm밖실행거절/문서링크검사PASS; Delta실제접속·제출은아직없다. 홈/아키텍처가달라기존ARMSIF를DeltaCPU에서그대로실행하지않는다.

- `/tmp/b6-c2-recompile`, branch`b6-c2-recompile-20261011`, **198ce14**, clean/pushed. state/journal/trainer통합·캐시경로결속독립검토PASS; d7b1a37은수정기준.
- 구현브랜치는 main과별개. `/tmp`가없으면브랜치에서worktree재생성. 현재main은업무기록/기존미커밋내용보존용이다.
- main운영도구: production7266e14, controller6e5f0f7, 연결시험ea699d5. 검증resumed410 제자리 활성화와 본학습 연결·최종 검수가 실제로 완료됐다. 새 controller journal은 complete; 옛 controller3276938은 계속 blocked_determinism/jobs[]이며 둘 다 재가동하지 않는다.
- 다음 게이트: B5 전체 preflight·checkpoint/재개 조건 확인, B6 검증된 소스/런처/입력 교체 검토 후 후속 실행. C2 완료 잡 재제출이나 canonical 경로로의 물리적 이동은 하지 않는다.
- 상세경로·검증결과·주의점·서버명·원격journal·잡ID는handoff에있다. 지금완료진단을다시돌리거나기존controller를그대로재가동하지않는다.

이전문서: [resume_history_20261012.md](resume_history_20261012.md).

원격보존: main스냅샷`8ef1871`과모든로컬작업브랜치푸시·SHA대조완료. [영수증](../notes/PUSH_RECEIPT_20261012.json). 최종문서tip은`git log -1`로확인.
