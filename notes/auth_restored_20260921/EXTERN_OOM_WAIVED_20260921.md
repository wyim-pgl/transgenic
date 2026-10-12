# `.extern OUT_OF_MEMORY 0:125` — 저자 결정으로 무시 (2026-09-21)

## 결정

> 저자: **"extern OOM 은 그냥 무시한다."**

`.extern OUT_OF_MEMORY 0:125`는 더 이상 **차단 조건이 아니다.** 원인 조사도 종료한다.

## 이 결정이 정당한 이유 (권한 관계)

이 게이트는 사용자·과학 spec이 만든 것이 아니라 **엔지니어링 독립검토가 세운 보수적 운영 수용
조건**이었다. 출처는 `notes/wait_work_20260916/longread/independent_review.md:11-17` — pilot
코드가 최종 검증보다 `COMPLETE` marker를 먼저 쓰는 실제 결함이 있어서, "batch/extern의
timeout·OOM·nonzero 거부"를 bounded pilot 승인 조건에 묶었다. `B6_MVP_PILOT_v1.md`에는
`extern`/`OOM`/`Slurm` 명칭의 조건이 없다(09-17 재점검에서 검색으로 확인).
게이트를 세운 주체가 저자가 아니므로 **저자가 해제할 수 있다.**

## 무엇이 해제되나

- `.extern`이 `OUT_OF_MEMORY 0:125`라는 이유만으로 잡 결과를 거부하지 않는다.
- ONT **CLEAN / F4 / full 실행**의 **회계 게이트는 해제**된다.
- 관리자 kernel·slurmstepd·cgroup 로그 확보는 **더 이상 선행조건이 아니다.**
- `notes/wait_work_20260916/repro/SLURM_EXTERN_SUPPORT_DRAFT.md`는 **미발송으로 종료**한다
  (발송하지 않으며, 기록으로만 보존).
- `.atl/todo.md`의 "extern 원인 조사" 항목을 종료한다.

## 무엇이 해제되지 않나 (중요)

09-17 재점검 자체가 이미 적어둔 문장이다: **"extern만 해결하면 full 실행 준비가 끝난다는 뜻이
아니다."** 이 결정도 마찬가지로 남은 조건을 건드리지 않는다.

1. **실제 승인도구 fixture 미검증.** job 6165965는 full wrapper를 **배포만 하고 실행하지 않았다**
   (`ont_full/wrapper.py` = `ont_fixture_v4/package/wrapper.py`, 바이트 동일 `314d6027…`).
   실행된 건 `fixture.py`였고 wrapper 11함수 중 6개만 도달, **`check_boundary`·`execute`·
   `fresh`·`pinned`·`slurm` 5개는 미실행**이었다. `slurm()`은 어떤 실제 allocation에서도 돈 적이 없다.
2. **round 1·2 수정본은 아직 실제 잡으로 검증되지 않았다.** `ont_full_v2`에서 커버리지
   15/15(`sys.settrace`)와 29 tests GREEN(3.10·3.12)을 얻었지만 전부 **로컬 합성**이다.
   로컬에는 승인 도구가 없다(minimap2 부재, samtools 1.13≠1.3.1, seqkit 2.0.0≠2.13.0).
3. **v5 패키지 미빌드** — 저자 지시로 게이트 대기 중이었다. 이제 빌드 자체는 막히지 않으나
   빌드·배포는 별도 실행안이다.
4. **throughput/scratch 예산 미측정.** `longread/REPORT.md:74-84`의 16CPU/32GiB/4h는 계획치이지
   측정치가 아니다.
5. **자원/중복 통제와 독립검토는 그대로 유효.** 이 결정은 회계 게이트 하나만 해제한다.

## 기록 원칙

`.extern` 상태는 **계속 receipt에 그대로 기록한다.** 관측을 지우는 것이 아니라 그것으로
**판정하지 않기로** 한 것이다. 과거 문서의 "externOOM 미해결"은 사실 진술로 유효하며,
바뀐 것은 그 사실의 **효력**이다.

## 이 결정의 무게 (FLNC 결정과의 연결)

A47.1로 **A-ONT1이 A. thaliana MVP의 유일한 구조 소스**가 됐다. 따라서 이 ONT 트랙이 구조
readout 전체를 혼자 떠받친다. 회계 게이트가 열렸다는 것은 **남은 장애물이 전부 1~4번이라는 뜻**이며,
그쪽이 해결되지 않으면 MVP는 여전히 INCONCLUSIVE로 간다.

## 범위

문서 기록만. 잡 제출·취소, 원격 쓰기, support 발송, v5 빌드, 커밋·푸시 없음.
GPU 잡 3187760과 B5 체인 3149358-63 무접촉.
