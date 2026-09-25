# 저자 결정 기록: 2026-09-25 저녁

기록 시각: 2026-09-25 16:40 PDT 전후. 결정은 모두 사용자 원문 그대로 남긴다.

## 1. ONT UMI: 125런 모두 absent 유지

- 원문: `동결된 TSV에는 125런 모두 absent 로 유지해`
- 적용: `data/manifests/ONT_PROTOCOL.b6_v1.draft.tsv`(sha256 `e21425e1…b029`)의 학습 대상 125런 UMI 열을 `absent`로 확정한다. TSV 바이트는 바꾸지 않는다.
- 근거 기록: [umi_research.md](umi_research.md)에는 표준 kit 근거가 있는 no_UMI 5런과 unresolved 120런이 있다. 이 결정은 unresolved 120런까지 저자 판단으로 `absent`를 채택한 것이며, 조사 결과 자체를 "UMI 미사용 확인"으로 바꾸지 않는다.
- 결과: C0에서 cDNA 83런은 PCR-equivalence 단위(`pcr_equivalence`), direct_RNA 42런은 한 read가 한 단위(`direct_rna`)가 된다(`b6_build_c0.py:776-783`).

## 2. Vvinifera EST library 누락 175 records: library = `NCBI-EST`

- 원문: `Vvinifera: 450,180개 중 175개 레코드(0.039%)에 library 정보 NCBI-EST`
- 적용: screened 450,180 records 중 GenBank에 `/clone_lib`가 없는 175 records(175 accessions)에 library 식별자 `NCBI-EST`를 부여한다. 다른 records와 다른 종의 library는 바꾸지 않는다.
- 이 값은 저자 지정 값이며 원본 GenBank에서 복구한 값이 아니다. 해당 records는 하나의 library로 묶인다.

## 3. extern OUT_OF_MEMORY 표시: 전역 무시

- 원문: `extern OOM 표시 는 전역에 무시 하라고 해`
- 적용: Slurm `.extern` 단계의 `OUT_OF_MEMORY 0:125` 표시는 모든 잡에서 판정에 쓰지 않는다. 잡 판정은 job/batch 상태·ExitCode·산출물 QC와 해시로 한다. batch 단계 OOM은 이 규칙에 해당하지 않는다.

## 4. 진행 방식

- 원문: `주말이라 멈추지 말고 계속 진행해줘`
- 적용: 승인된 범위의 실데이터 준비를 확인 요청 없이 계속한다. 새 저자 결정이 필요한 과학적 판단은 기록하고 다른 실행 가능한 항목을 진행한다.
