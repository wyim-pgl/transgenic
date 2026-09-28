# Intent: A40 코퍼스로 B5 학습 런 시작

> **재시작 갱신 — 2026-09-28 14:30 PDT:** 현재 목표는 **실제 생물학 데이터로 B6 학습 입력(증거 DB·코퍼스·cache·bindings)을 만들어 WP5 4-GPU 파일럿(B6 vs C2, 4,099 optimizer updates)을 제출하는 것**이다. 사용자가 운영학습 시작을 승인했고(09-25), 과학적 판단은 저자 결정 1–18로 확정했다([결정 원문](../notes/b6_production_start_20260925/AUTHOR_DECISIONS_20260925_evening.md)). 파일럿 GPU 제출 직전에는 예산과 함께 사용자 확인을 받는다. 실행 정본은 [resume.md](resume.md)와 [RESTART_HANDOFF_20260928](../notes/RESTART_HANDOFF_20260928.md)다. 아래 B5·B6 초기 서술은 역사 기록이다.
>
> 시작 전 참이어야 하는 것(현재 상태): ☑ 모든 종 C0 exhaustiveness ≥0.999(동일 audit N, Athaliana·Osativa ONT 최종 확인 대기) · ☑ B6 코퍼스 repair·검증(행 196,774) · ☐ protein C0 9종(6/9) · ☐ ONT C0(4/6) · ☐ zero list·병합·cache·bindings · ☐ DeltaAI 전송·해시 대조 · ☐ 사용자 GPU 제출 확인.

> **재시작 갱신 — 2026-09-23 10:15 PDT:** B5 학습 intent의 역사와 근거는 보존한다. 현재 진행은 **B6 증거 모드**다. A46 v1.36은 2026-09-22에 승인됐고(A48 v1.40 추가), WP1(C0 증거 DB) 코드는 `/tmp/b6-wp1-20260922`에서 개발 중이며 커밋되지 않았다. 실행 정본은 [resume.md](resume.md)와 [RESTART_HANDOFF_20260923](../notes/RESTART_HANDOFF_20260923.md)다. B6의 목적과 사전 조건은 A46 본문과 [B6 워크플랜](../revision/protocols/B6_EVIDENCE_MODE_WORKPLAN_v1.md)이 정한다. 아래 B5 상태 서술은 현재 실측이 아니다.

> **재시작 갱신 — 2026-09-15 15:13 PDT:** 이 파일의 B5 역사/근거는 보존한다. 현재 실행 정본은 [resume.md](resume.md)와 [상세 인수인계](../notes/RESTART_HANDOFF_20260915.md). 저자는 EST blastx·MVP·격리 worktree·독립 병행을 승인했고 blastx6164049 및 full MVP DB6164118이 실행 중이다. MVP data/model/dataset 코드는 c7fa759에 보존되어 있으나 trainer/GPU학습은 미구현/미실행. **paired 검증손실을 순수 weighted CE로 정의할지 아직 저자 결정 대기.** A46 전체 동결과는 별개 승인이다. DeltaAI 인증 실패로 seed123 현재 상태는 확인하지 못했다. 아래의 epoch/잔액/큐 상태를 현재 실측으로 인용하지 않는다.

**Author:** Won Yim (PGL) · **Status:** draft — 저자 수락 대기 · **작성** 2026-09-07 · **갱신** ✏️ 2026-09-14 11:5x PDT(말미 "후속" 절: intent 실행 중 — 시드 123 epoch 4·11 h 링크 체인, 456/789 연기; B6는 별도 워크플랜 + A46) · 2026-09-10 오전 PDT (R1–R6 ☑; **R7 ☑ 25.06·26.08**(`3124099`·`3124229`); **R8 ☑ 저자 결정 26.08, A44 v1.34**; ✏️ 12:1x PDT: `3124347`·`3124348` PASS → R7 26.08 완전 ☑; epoch 체크포인트 재개 인증 `3124347` 대기; ~~2026-09-09 저녁: R7 ☑ 25.06 / ✏️ 26.08 중단 경로만 미인증; R8 ☐ 저자 결정~~; A28 체인 리허설 실증; ~~R7 ☐ 진행 중~~; SEED=123 학습 중; ~~compile 불가~~ → **NGC 26.08에서 compile 통과, 25.06/26.04는 불가**; 확인 시드 4 GPU 결정)
**Source:** [The AI-Native SDLC Playbook](https://claude.com/blog/the-ai-native-sdlc-playbook) — Plan 단계 산출물

> 📌 **정본 구분**: 지금 상태는 [`resume.md`](resume.md) 📌 블록 · 다음 행동은 [`todo.md`](todo.md) ·
> 쓰지 않기로 한 것은 [`quarantine.md`](quarantine.md) · 진행 상태의 정본은 GitHub 이슈
> wyim-pgl/transgenic. **이 파일은 "왜 이걸 하려는가"와 "시작 전 무엇이 참이어야 하는가"만 담는다.**
>
> intent는 **결정이 아니라 제안**이다. 여기 적힌 것은 저자가 수락하기 전까지 구속력이 없다.
>
> ⚠️ **상태는 실측 명령과 그 출력으로만 적는다.** 요구사항은 `☐/☑`가 붙은 readiness gate 행으로만
> 적는다. 이 규칙과 아래 세 판정 절차는 [`quarantine.md` §1g](quarantine.md)에서 왔다 — 2026-09-07에
> 준비 상태를 점검하며 **문서가 틀렸다고 세 번 적었다가 세 번 다 내 쪽 결함으로 정정했다**:
>
> 1. **grep 한 줄로 문서를 판정하지 않는다.** 괄호 문장의 선행사는 항목 제목까지 읽어야 정해진다.
> 2. **"코드가 없다"고 판정하기 전에 `git log --oneline -1`로 저장소를 찍는다.** 저장소가 둘이고
>    디렉터리 구조가 같아 겉보기로 구분되지 않는다.
> 3. **외부 검토자에게는 근거 파일을 다 준다.** 부분집합을 주면 돌아오는 것은 코드 판단이 아니라
>    내가 만든 결핍의 반사다.
> 4. **축이 둘인 비용을 한 축만 재고 나머지가 평평하다고 쓰지 않는다.** 2026-09-08에 단백질 정렬을
>    게놈 크기만 바꾸고 질의 수는 10만으로 고정해 사이징했고, 드라이버 주석에 "메모리는 질의 수가
>    아니라 인덱스가 지배한다"고 **측정 없이** 적었다. 죽은 것은 질의 수 축이었다(OOM 2건).
>    측정하지 않은 축은 "평평하다"가 아니라 **"모른다"**로 적는다.

---

## Problem — 지금 못 하는 것

B5 학습을 시작할 수 없었다. 막고 있던 것은 자원이 아니라 **코퍼스가 자기 계약을 이행하지 않는다는 사실**이었다.

`b5_full_v1`은 세 조항을 지키지 않았다(#55·#56·#58). *V. vinifera* no-CDS 좌위 6,205개가 마스킹·overlap
closure 이전에 제거돼 valid 634 · test 971개가 A29/A33.3 보호를 우회할 수 있었고, 빈 라벨은 3토큰을
저장하면서 디코더 tokenizer는 2토큰만 방출했으며, emitter는 annotation span으로 정렬하는데 validator는
emitted feature로 정렬해 forward window 97건이 canonical order로 거절됐다.

**2026-09-06/07에 이 셋은 해소됐다.** 교체 코퍼스 `b5_full_a40_v1`(273,148행, A40/v1.30)이 동결·채택됐고
`f6e68d4`·`5289077`로 커밋됐다.

**그런데 코퍼스가 준비됐다는 것과 학습이 시작될 수 있다는 것은 다르다.** 2026-09-07 실측에서
ACCESS 번들과 실행 스크립트가 **여전히 폐기된 코퍼스를 가리키고 있음**이 드러났다. 아래 readiness
gate가 그것이다.

## Proposed outcome — 나아진 상태

ACCESS 계정이 발급되는 **그 날 첫 시드가 큐에 들어간다.** 자원 이름만 채우면 되는 상태여야 하고,
그 외의 어떤 것도 그날 발견되어서는 안 된다.

하나의 런 라벨이 다음을 한꺼번에 고정한다.

- §1 객체 `b5_full_a40_v1.db` — MD5 `dda7e78c880c993435654b57dcd1c498`, 273,148행, 23,534,252,032 B
- 수정된 3토큰 empty-target tokenizer (`<s> <empty> </s>`)
- `#59` seedable sampler — 리비전 저장소 `1413efd`에 **커밋돼 있다**(실측)
- 프로토콜 A40 / v1.30

성공 판정은 "학습이 끝났다"가 아니라 **"체크포인트에서 재개해도 같은 런임을 증명할 수 있다"**이다.
seed 하나가 GH200 1장 기준 4–6일이고 잡당 벽시계 상한이 48시간이므로, 재개 사슬이 정체성을 보존하지
못하면 런 자체가 성립하지 않는다.

---

## Readiness gate — ACCESS 전에 참으로 만들어야 하는 것

각 행은 **실측 명령**과 **현재 출력**을 함께 적는다. ☐는 아직 거짓, ☑는 실측으로 참.

### R1 ☑ 번들의 DB가 A40 객체다 (2026-09-07 해소)

```
$ ls -l ~/scratch/data/Transgenic/training_input/db/
b5_full_a40_v1.db   23,534,252,032   md5 dda7e78c880c993435654b57dcd1c498   ✅
```
gpu에서 rsync 후 md5·sha256·bytes·행수를 freeze와 대조했다. 폐기 `b5_full_v1.db`는 삭제가 아니라
`RETIRED_DO_NOT_USE/training_input_db_b5_full_v1_20260907/`로 **경로 격리**했다(`WHY_RETIRED.txt` 동봉).
원본은 gpu에 보존돼 있다. 이하는 해소 전 기록이다:

```
b5_full_v1.db   23,491,260,416      ← 폐기된 코퍼스 (2026-09-07 격리됨)
```

번들은 **superseded `b5_full_v1`을 담고 있다**. A40 객체는 `gpu:/home/pgl/scratch1/wyim/transgenic_data/b5/a40_20260905_v1/merged/b5_full_a40_v1.db`(23,534,252,032 B)다.
이 상태로 rsync하면 **A28이 금지하는 "폐기 데이터 사슬이 조용히 A40 런이 되는 것"**이 정확히 일어난다.

**해야 할 것**: A40 DB를 번들로 옮기고 md5를 freeze와 대조, sha256 사이드카 재생성, 구 DB는 검증 후 제거.
**판정**: 번들 DB의 md5 == `dda7e78c880c993435654b57dcd1c498`.

### R2 ☑ 번들이 자족적이다 (2026-09-07 해소 — 번들 안에서 실체화)

```
$ find ~/scratch/data/Transgenic/training_input -type l | wc -l
9        ← genomes/*.fa 전부, /data/gpfs/assoc/pgl/data/Transgenic/genomes/ 를 가리킴 (9/9 대상 존재)
```

**이건 결함이 아니다.** 번들 README가 명시하고 있다 — "`genomes/` — 학습 9종만.
`Transgenic/genomes/`로의 심볼릭 링크이며, 표에는 실경로와 md5가 들어간다." `MANIFEST.tsv`가 실경로와
md5를 담으므로 링크여도 내용 정체성은 보존된다. 번들의 나머지(게이트 스크립트 포함)는 전부 실파일이다.

**정해지지 않았던 것은 전송이었다.** ACCESS에는 `/data/gpfs/assoc/pgl/...` 경로가 없으므로 링크를 그대로
rsync하면 dangling으로 도착한다.

**2026-09-07 결정: 번들 안에서 실체화한다.** `rsync -L`은 **전송 명령을 정본으로 만든다** — 그 옵션을
빠뜨리는 날 조용히 깨지고, 깨진 것은 ACCESS 쪽에서야 드러난다. 번들이 자기 완결적이면 그럴 자리가
없다. 비용은 4.1 GB이고 여유는 28 TB다.

```
$ find ~/scratch/data/Transgenic/training_input -type l | wc -l
0
$ # MANIFEST.tsv 대조: 9/9 md5 일치, mismatch=0
```

실체화 후 `MANIFEST.tsv`를 재생성했고, 생성기의 낡은 설명 두 개도 함께 고쳤다 —
`db` 행이 `272,224 rows`(구 코퍼스 행수)라고, `genomes` 행이 아직 `symlink to`라고 적고 있었다.
md5와 경로는 맞는데 **설명만 틀린 형태**라 다음 사람이 그것을 읽는다.

### R3 ☑ 실행 스크립트가 A40 객체를 가리킨다 (2026-09-07 해소)

```
deploy/deltaai/train_b5.slurm:32   sha256sum -c /work/db/b5_v1.duckdb.sha256
deploy/deltaai/train_b5.slurm:35   --db /work/db/b5_v1.duckdb
```

폐기된 이름이 **하드코딩**돼 있다. 이름이 낡았을 뿐 아니라, sha256 사이드카 검사가 있으므로
파일만 바꾸면 **게이트가 정상적으로 실패한다**(fail-closed는 작동한다 — 조용히 틀리지는 않는다).
**판정**: `--db` 경로와 sha256 사이드카가 A40 객체를 가리키고, 사이드카 값이 freeze와 일치.

### R4 ☑ ACCESS 자원명·계정·파티션 (2026-09-08 해소)

✏️ **해소**: NCSA **DeltaAI**, 계정 `bilv-dtai-gh`, 파티션 `ghx4`(MaxTime 48 h, GH200 120 GB×4), 사용자 `wcyim`, 저장소 `/work/nvme/bilv/wcyim`. 키 인증은 정책상 불가 → 사용자가 pronghorn `login-0`에서 `ssh -fN dtai`로 ControlMaster를 열고 에이전트는 소켓에 얹힌다(resume.md 참조). 예산 999 GPU-h(추가 배정 가능). 과금 `MAX_TRES` → `--mem`을 쓰지 않는다(A42). 아래는 해소 전 서술.


```
deploy/deltaai/train_b5.slurm:3-4  #SBATCH --account=CHANGE_ME
                                   #SBATCH --partition=CHANGE_ME
```

**우리가 풀 수 없는 유일한 항목.** open question 1과 같은 것이다. 나머지 R1–R3는 지금 끝낼 수 있다.

### R5 ☑ 첫 시드 전 preflight가 한 명령으로 돈다 (2026-09-07 해소)

```
$ bash deploy/deltaai/preflight_b5.sh --bundle <training_input> --repo <repo>
PREFLIGHT PASS — this run would consume the A40 object the freeze names
```

`deploy/deltaai/preflight_b5.sh`. 기대값은 **freeze JSON에서 읽는다** — 하드코딩하면 freeze와 스크립트가
따로 낡는다. `train_b5.slurm`의 `sha256sum -c` 자리를 이것이 대체한다.

⚠️ **이 스크립트 자신이 첫 실행에서 허위 실패를 냈다** — tokenizer 검사가 `datasets.py`를 grep 했는데
`<empty>`는 `model/tokenization_transgenic.py`에 있다. 지금은 **실제로 토큰화해서 길이를 잰다**(임포트가
안 되는 환경에서는 정본 두 파일의 선언으로 낮춘 검사를 하고 **그 사실을 출력에 밝힌다**).

R1–R3를 사람이 기억해서 확인하는 한 반복 실수는 계속된다. **첫 optimizer step 이전에** DB md5·
tokenizer 3토큰·seedable sampler·번들 자족성을 한 번에 검사하고 하나라도 어긋나면 exit ≠ 0 하는
스크립트가 있어야 한다. `#69`(게이트가 opt-in이라 생략하면 조용히 낡은 신뢰로 돌아감)와 같은 계열의
문제다 — **검사가 선택적이면 언젠가 생략된다.**

### R6 ☑ #51 벤치마크 (2026-09-08 실행 — #18에 게시)

✏️ **해소**: `bench_b5.slurm` 3티어 × 1 GPU(eager; compile은 HF checkpointing과 충돌해 폴백): 30/61/129 kb = **24.9 / 45.5 / 96.6 s/step**, peak **101 GB 세 티어 동일**(8,192-토큰 디코더가 지배, A27) → micro-batch 1 강제. 실제 시드 에폭(티어 혼합) **≈ 20 h**. 4-GPU 런은 DDP 결함(`af6cce8`) + **유효배치 96×N 버그**(`2021fd2`)로 실패 → 2-GPU + non-reentrant ckpt로 **17.57 s/step(1.42×)** 확보. 기본값 `STEPS=300`은 2 h 한도 초과라 `STEPS=50`. 아래는 실행 전 서술.


`deploy/deltaai/bench_b5.slurm`은 존재한다. 129,024 nt 계층은 4090에서 물리적으로 측정 불가라
(1,093/1,103 OOM) **ACCESS에서만 확정된다.** 이것이 크레딧 요청의 근거가 되므로 학습보다 먼저 돈다.

✏️ **2026-09-09 낮**: 위 "compile은 HF checkpointing과 충돌해 폴백"은 **25.06·26.04에 한한다** — NGC 26.08(torch 2.14.0a0)에서는 whole-model compile + checkpointing이 30 kb·129 kb 모두 통과했다(3 step probe, OOM 없음). 정상 상태 s/step·호스트 peak·4 GPU 스케일링은 벤치 매트릭스 `3118903`–`3118908`(50 step)이 측정 중이며, 이 값이 시드 사이징의 정본이 된다. ⚠️ 위 "peak 101 GB"와 26.08의 "11~24 GB"는 `max_memory_allocated`가 torch 판 간에 다르게 세는 것으로 의심돼 **비교 금지** — `3118909`(25.06 eager + nvidia-smi)가 판정한다.

### R7 ☑ (25.06 · 26.08) DDP 경로가 체크포인트·재개·검증·중단을 실제로 통과한다 (2026-09-09 낮 신설, 저녁 25.06 해소, 09-10 오전 26.08 해소)

**왜 필요한가.** 2 GPU 벤치(3117590·round 6)는 50 step이라 `save_every 200`·epoch 끝에 닿은 적이 없다. Codex 리뷰(2026-09-09)가 그 경로에서 결함 3건을 찾았고(`5bc4df8`에서 수정: rank-0 디렉터리 소유 + barrier, 검증/학습 손실·USR1 플래그 합산, DDP KeyboardInterrupt 저장 금지), 수정은 소스 검증뿐이다.

**판정 명령(실측).** `deploy/deltaai/ddp_integration.slurm` — 4 GPU, 480행 30 kb 부분 DB, save-every 2, rank 하나에만 USR1, `--resume auto`, `--max-epochs 2 --patience 1`:
```bash
ssh dtai 'grep -h "DDP_INTEGRATION\|leg[12] rc" /work/nvme/bilv/wcyim/transgenic/repo_dev/ddp-int-3118861.out /work/nvme/bilv/wcyim/transgenic/repo_dev/ddp-int-2608-3118880.out'
```
☑ 조건: 두 잡 모두 `DDP_INTEGRATION PASS`(25.06 eager `3118861`, 26.08 compile `3118880`). ~~현재 **PENDING**~~

✏️ **2026-09-09 저녁**: **25.06 ☑** — `3119681`(기본 할당기)·`3119682`(`ALLOC=native`) **PASS**(rank-0 저장, rank 하나 USR1 → 4 rank 함께 종료, latest_state 재개, 검증 합산, TRAINING_DONE). `3118861`은 step-0 launch failure로 leg 1 사망, `3118880` 취소. **26.08 compile ✏️ PARTIAL** — `3119683`/`3119779`는 학습·저장·검증·재개·TRAINING_DONE 통과, **USR1 중단만 미발화**(dynamo가 SIGUSR1 핸들러를 막음; SIGTERM은 됨 — quarantine §1j) → 시험 스크립트를 SIGTERM으로 바꿔 재실행해야 ☑. 판정 명령의 잡 번호는 `3119681`/`3119682`/재실행분으로 읽을 것.

**같은 날 드러난 상위 조건 — A28 체인 자체.** 48 h 경계의 USR1 전달은 오늘 전까지 실증된 적이 없었고 결함 6건이 있었다(resume 저녁 블록). 리허설 `3120089`(정상 경로)·`3120124`(강제 선점 경로) 통과로 이 조건은 닫혔다; 확인 시드는 이 스크립트(`1a245ce`)로 제출한다.

✏️ **2026-09-10 오전**: **26.08 ☑** — `3124099`(26.08 compile, 4 GPU, rank 하나에 **TERM** → 4 rank 저장·종료·재개·TRAINING_DONE) PASS; 어제 미발화는 USR1 때문이었고(dynamo) 통합 시험도 체인처럼 TERM을 보낸다(`35b1436`). **A28 체인도 26.08·4 GPU·`repo_seeds`로 `3124229` PASS**(COMPLETED 0:0, GPU별 워밍업·preflight·`runtime identity`·compile·SIGTERM → 전원 저장·종료·chain limit). 남은 인증: `3124347`(`KILL_AFTER_EPOCH=1`, `NTRAIN=484` — epoch 체크포인트 재개가 accumulation 잔여분을 복원하는지: `restored … pending` + `step 23 global_step 6` + `pending_micro: 2`)와 `3124348`(정상 경로, `c646e76`). 이 둘이 PASS면 R7은 확인 시드 실행 코드(`c646e76`)에 대해 닫힌다.

### R8 ☑ 확인 시드 456/789의 실행 환경이 결정·기록됐다 (2026-09-09 낮 신설 — 2026-09-10 저자 결정 **NGC 26.08**)

**갈림.** (i) 25.06 eager — 시드 123과 이미지·torch·커널이 같다; 4 GPU DDP만 다르다(rank당 acc 24, reduction 순서). (ii) 26.08 compile — 벽시계 이득(크기는 R6 ✏️ 벤치가 정함)이 있지만 torch 2.8→2.14, transformers/accelerate 동일(4.57.6/1.15.0), compile 커널이 시드 123과 다르다 → **개정 A44**로 "확인 시드는 다른 이미지에서 돈다"를 명시해야 한다. 이 파일은 제안이지 결정이 아니다 — 결정은 저자.
☑ 조건: 결정이 `todo.md` 저자 결정 (b)에서 닫히고, (ii)면 A44가 커밋됨.

✏️ **2026-09-10 오전**: **(ii) 결정** — 저자 "NGC 26.08 사용한다". A44(v1.34, `c646e76`) 커밋: 이미지 sha256 `1a5dee27…`·torch 2.14.0a0·compile·4 GPU DDP(rank당 acc 24)·non-reentrant ckpt·`repo_seeds` c646e76; 시드 123은 25.06(`3b242b33…`)·eager·1 GPU·08f3e2b 그대로. **비용을 문서에 명시**: 세 시드는 시드만 다른 재현이 아니라 혼합 런타임 재현이다(reduction 순서·컴파일러 커널·RNG 스트림) — 중립성 주장 없음, 비교는 검증 손실 궤적·종료 epoch·하류 평가. 각 런의 첫 링크가 `runtime_identity.txt`를 남긴다. 제안이 아니라 결정이므로 이 항목은 닫힘.

---

## Affected users and systems

| 대상 | 영향 |
|---|---|
| **ACCESS (DeltaAI 가정)** | 첫 시드가 여기서 돈다. `deploy/deltaai/`가 이미 DeltaAI를 가정하고 있다 — A100 80 GB로 확정되면 이 경로 이름부터 재검토해야 한다 |
| **pgl-gpu** | A40 §1 DB 원본 보관처. 번들 갱신의 출발지 |
| **pronghorn** | 리비전 저장소(**학습 코드 정본**)와 증거. 학습은 여기서 돌지 않는다 |
| **원고 Methods** | "partially pretrained"(A34) 미반영. #61 결정에 따라 test set 정의 문장이 갈린다 |
| **`b5_full_v1` 사용처** | 없다 — 학습을 시작한 적이 없어 무효화되는 하류 산출물이 없다 |

## Constraints — 경계

- **A28**: 기존 잡 체인 안에서 코드·데이터를 패치하지 않는다. 레시피가 바뀌었으므로 **새 labelled run**이어야
  하고, superseded 데이터/tokenizer 사슬이 조용히 A40 런이 될 수 없다. **R1·R3가 정확히 이 조항의 대상이다.**
- **§§3–9 불변**: 첫 결과표가 나온 뒤에는 §§3–9를 못 바꾼다.
- **48시간 벽시계 상한**: seed 하나가 한 잡에 안 들어간다. `train_b5.slurm`이 SIGUSR1(종료 15분 전)에
  `latest_state`를 저장하고 `--dependency=afterany`로 자기를 재제출한다(`CHAIN_MAX` 기본 8).
- **동결 무결성**: `b5_full_a40_v1.freeze.json`이 `scripts/merge_b5_databases.py`와
  `tests/test_merge_b5_databases.py`를 SHA256으로 고정한다. 이 둘을 고치는 변경은 **별도 커밋**으로만
  들어간다(`f6e68d4` / `5289077`로 나눈 이유).
- **저장소가 둘이다.** 학습 코드 정본은 **`pronghorn:~/scratch/data/Transgenic/transgenic`**이고,
  `gpu:/home/wyim/transgenic`은 낡은 클론이다(HEAD `e36f051`). 두 저장소 모두 `deploy/deltaai/`와
  `revision/`을 가지고 있어 **겉보기로 구분되지 않는다.** 코드가 없다고 판단하기 전에 **어느 저장소를
  보고 있는지 먼저 확인한다** — `#59`가 "미구현"으로 보였던 것이 이 착시였다([`quarantine.md` §1g](quarantine.md)).
- **테스트는 컨테이너에서 돈다.** 로그인·컴퓨트 노드 모두 `duckdb` import가 `GLIBC_2.23 not found`로
  실패한다. `singularity exec -B /data/gpfs ~/scratch/singularity/micromamba.sif /bin/micromamba run -r ~/micromamba -n base python -m pytest -q`
- **외부 검토자에게는 근거 파일을 다 준다.** 부분집합을 주고 "근거가 충분한가"를 물으면 돌아오는 것은
  코드 판단이 아니라 **내가 만든 결핍의 반사**다(A40 검토에서 허위 블로커 3건, [`quarantine.md` §1g](quarantine.md)).
- **자원 사이징은 실제로 변하는 축에서 잰다.** 파일럿이 축 하나를 고정하면 그 축에 대해서는 아무것도
  측정하지 않은 것이다. B5 학습에서 이 축은 **컨텍스트 계층 × 스텝 수**이고, #51 벤치마크가 계층은
  재지만 긴 런에서의 누적은 재지 않는다 — 첫 시드를 길게 돌리기 전에 **메모리가 스텝에 따라 평평한지**를
  확인해야 한다. 2026-09-08 단백질 정렬 OOM이 이 조항의 근거다(`resume.md` 2026-09-08 스냅샷).
- **커밋 규약**: 저자 Won Yim <wyim@unr.edu> 단독.

## Open questions — 수락 전 정해야 하는 것

✏️ **2026-09-09**: **1(자원 사이징)은 닫혔다** — 실측: 1 GPU eager 에폭 ≈ 20 h(티어 혼합, 2.4~2.9 it/s), 시드 8~10 에폭 ≈ 7~9일 ≈ 170~220 GPU-h(48 h 링크 4~5개); 2 GPU DDP(non-reentrant ckpt) 1.42×; compile 불가(129 kb+ckpt 필수). 남은 열린 질문은 **확인 시드에 DDP를 쓸지**(numerics 미세 차이)와 **4-GPU 스케일링 미측정**. `spec.md`를 쓸 수 있다.

✏️ **2026-09-09 낮**: ~~확인 시드에 DDP를 쓸지~~ → 저자 결정 **4 GPU DDP**(예산은 제약 아님, 저자가 보충). ⚠️ 위 "에폭 ≈ 20 h"는 13:10 실측(46 %/12 h)으로 **≈ 26 h**. 새 열린 질문 **7: 확인 시드 이미지**(25.06 eager vs 26.08 compile, R8) — 벤치 매트릭스와 R7 결과 뒤 저자가 정한다. 4 GPU 스케일링은 `3118904/6/8`이 측정 중.

✏️ **2026-09-10 오전**: 질문 7(이미지) **닫힘 — 26.08**(R8). 4 GPU 스케일링 실측 91~93 %(epoch 6.15 h). 열린 것: **8** 트레이너 파일 기반 stop 플래그(SIGTERM이 되므로 급하지 않음) · **9(신규)** 시드 123의 `train_epoch_loss`(epoch ≥ 2)가 누적 카운터로 나뉘어 epoch 번호만큼 과소 — 결과표에서 어떻게 보정·표기할지(검증 손실 무관) · **10(신규)** 시드 123 동결 트레이너는 epoch 체크포인트에 accumulation 잔여분(12 micro-batch)을 남기지 않는다 — epoch 체크포인트가 최신 복구점이 되는 일이 생기면 런 로그에 편차 기록(A44 3항) · pending_grads rank-0-only 최적화(선택).


| # | 질문 | 갈리는 것 | 정본 |
|---|---|---|---|
| 1 | **ACCESS 자원명·계정·파티션** | A100 80 GB vs GH200 96 GB — 벤치마크·크레딧 환산·계층 상한이 달라진다. **R4를 막고 있는 유일한 것** | #51·#19·#20 |
| 2 | **#61 논문의 "test set" 정의** | 명목 orthogroup split(91,787) vs 실제 라벨된 16.3 %(15,005). Methods 문장이 갈린다. A40은 이걸 정하지 않았다 | #61 |
| ~~3~~ | ~~**번들 자족성 방식**~~ | ✅ **2026-09-07 결정: 번들 안에서 실체화.** `rsync -L`은 전송 명령을 정본으로 만들고, 그 옵션을 빠뜨린 것은 ACCESS 쪽에서야 드러난다 | R2 |
| 4 | **§3.3 항목 1 — 정렬에 쓴 FASTA가 §1 객체가 아니다** | A. thaliana `513e5ef3` vs §1 `987ca803`. "헤더 차이일 뿐 서열 동일"은 2026-09-04 **진술이지 측정이 아니다** | todo.md §3 |
| 5 | **AtRTD3를 "깨끗한 정답"으로 쓸 수 있는가** | AtRTD3 자체 Iso-Seq에 `upf1`·`upf3`·`xrn2/3/4` 변이체와 핵분리 RNA가 포함돼 있다. **우리 검증 기준을 우리 정답이 통과하지 못한다** | todo.md §3 |
| 6 | **커밋 푸시 여부** | `f6e68d4`·`5289077` 미푸시. 푸시해야 #55·#56·#58이 닫힌다 | — |

**1은 ACCESS 시작 전 필수**(우리가 못 푼다 — 남은 유일한 외부 블로커). **2는 첫 결과표 이전에** —
결과를 보고 정의를 고르는 모양이 되면 안 된다. **3은 해소됐다.** 4–6은 병행 가능.

## 다음 산출물

플레이북의 사슬은 intent → spec → plan → diff → PR이다.

1. **R1·R2·R3·R5 실행** — open question 1과 무관하게 지금 가능
2. **`spec.md`** — open question 1·3이 닫힌 뒤. 런 레시피·체크포인트 사슬·성공 판정을 조항으로
3. **`plan.md`** — 바뀌는 파일, 작업 순서, 위험, 증명 방법
4. R6(#51 벤치마크) → #19 seed 123 → #20 seeds 456·789

✏️ **2026-09-09 현재**: **R1–R6 전부 해소.** SEED=123(잡 3116844)이 DeltaAI에서 1 GPU eager로 학습 중(A28 체인). 다음 산출물은 `spec.md`(가능해짐) → 확인 시드 456/789(DDP 여부 저자 결정) → #20. 이전 서술: R1·R2·R3·R5 해소, preflight PASS. R4(계정)가 유일한 외부 블로커.
병행 작업인 #45 단백질 정렬은 진행 중이며 OOM 2건을 겪었다 — B5와 자원이 다르므로 학습 시작을 막지 않으나,
위 "자원 사이징" 조항은 그 사건에서 나왔다.

✏️ 2026-09-09 14:0x: 첫 정상 상태 값 — 26.08 compile 30 kb 1 GPU **21.35 s/step (1.17×)**, ~~드라이버 peak 31.5 GB(torch 101 GB, 계정 의심; `3118909` 판정 대기)~~ ⚠️ CAUTION (저녁): torch 101 GB가 실제 순간 peak, nvidia-smi 2 s 샘플은 하한(§1j). 티어별·4 GPU 값이 모여야 사이징이 된다.

✏️ 2026-09-09 저녁: 1 GPU 26.08 compile 3티어 완료 — 21.35/40.54/86.64 s/step(1.17/1.12/1.12×), **1 GPU 에폭 ≈ 22.6 h**(eager 25.6 h). 오늘의 25.06 eager 대조(31.63 non-reentrant·29.96 reentrant)는 09-08의 24.9보다 20~27 % 느려 노드/일자 변동이 있다 — 사이징은 보수적으로 eager 값을 쓴다. 4 GPU 스케일링(`3118904/6/8`)이 아직 없어 시드당 GPU-h는 미정. 새 열린 질문 **8: 트레이너 파일 기반 stop 플래그**(`STOP_REQUESTED`, 수치 무관) — 체인이 시그널 전달에 의존하지 않게; 저자 결정 (c).

✏️ 2026-09-10 오전: R7·R8 닫힘(26.08). 순서: ~~`3124347`·`3124348` PASS~~(✏️ 12:1x PDT 둘 다 PASS 완료) → 시드 123 경계(21:45~22:00 PDT) 통과 확인 → ✏️ 시드 456은 `auto_submit_456.sh` 자동 제출(저자 선택지 2), 789는 저자 결정 뒤 제출(명령은 `todo.md` 상단) → `spec.md`(사이징: 4 GPU 26.08 epoch 6.15 h·24.6 GPU-h, 시드 197~246 GPU-h; 체크포인트 사슬: A28 + A44 3항; 성공 판정: patience 3) → #20.

✏️ 2026-09-09 낮: `spec.md`의 사이징 절은 **벤치 매트릭스(`3118903`–`3118909`) 값이 나온 뒤** 쓴다 — 3 step probe 값(45.2/110.0 s/step)은 compile 워밍업이 섞여 정상 상태가 아니다. 순서: R7·R8 → A44(필요 시) → `spec.md` → 시드 456/789(4 GPU).

✏️ 2026-09-09: open question 1이 닫혔으므로 이 이유는 소멸 — **`spec.md`를 쓸 차례다.** 사이징 절은 실측(위)으로, 체크포인트 사슬은 A28(`af6cce8` wait 루프 수정 포함), 성공 판정은 A25·patience 3. (이전: **`spec.md`를 아직 쓰지 않는 이유**: open question 1이 열려 있는 동안 쓰면 자원 가정을 지어내게 된다.)

## ✏️ 2026-09-14 — 이 intent의 상태와 후속(B6)

- **실행 중**: R1–R8 전부 ☑. 시드 123(잡 체인 `3131536` → `3131537` … `3149363`, 26.08·4 GPU·`repo_seeds` `8f049eb`, A45)이 epoch 4까지 완료(eval 0.8707 → 0.7029 → 0.6724 → 0.6525, best epoch 4), 링크 2 대기(48 h → 11 h 링크로 전환, 저자 지시 09-14). 시드 456/789는 저자 지시(09-12)로 **연기**(생략 아님, 프로토콜 273행). `spec.md`는 아직 없다.
- **열린 질문 갱신**: 2(#61 test set 정의)·4(§3.3 FASTA 동일성)·5(AtRTD3)는 여전히 열림; 8(stop 플래그)·9(train loss 보정)는 `todo.md` 이월 항목.
- **후속 intent = B6 증거 모드** — 별도 intent 파일을 만들지 않는다. "왜"는 `revision/protocols/B6_WORKPLAN_README.md` §1, 동결 정의·게이트는 `B6_EVIDENCE_MODE_WORKPLAN_v1.md` v1.2 §2 + A46 v1.36 초안(`PROTOCOL_B1_frozen_v1.md`, 미커밋), 1주 탐색 pilot은 `B6_MVP_PILOT_v1.md`(미커밋). B6의 readiness gate는 A46 15항(저자 승인·동결 전 WP1 착수 금지)이 대신한다. 이 파일(B5 런 시작)은 시드 123 완료와 456/789 재제출 뒤에 닫는다.
