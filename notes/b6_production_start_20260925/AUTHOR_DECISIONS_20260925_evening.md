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

## 5–8. 추가 결정 (2026-09-25 밤)

사용자 원문: `1. A 2. 추가 3. 괜ㅊ낳아 4. 별로로 정렬`

5. **HSP 투영 BTOP `X` = (a) 채택.** DIAMOND 기본 masking이 BTOP에 남긴 `X`는 masking된 자리로 인정한다. query `X`는 실제 codon이 정상 sense codon인지, subject `X`는 실제 잔기가 존재하는지만 검증한다. 다른 불일치는 계속 실패로 처리한다. blastx는 재실행하지 않는다.
6. **protein C0용 DATASET_ROLES `protein/<Sp>` 행 추가 승인.** 기존 동결 파일(`inputs/DATASET_ROLES.longread_v1.tsv`)은 실행 중인 잡이 쓰므로 바이트를 바꾸지 않고, protein 행을 더한 새 파일을 만들어 protein C0에만 쓴다.
7. **Athaliana cui2020 기존 정렬 재사용 승인.** 두 사본의 내용이 같으므로(gzip 시각 2바이트만 다름) 정본 dataset `ont/Athaliana_cui2020_PRJNA594286` 이름의 뷰로 기존 primary·audit 정렬을 쓴다. PROVENANCE·BAM header에 제외 사본 경로가 남는 것을 수용한다. metadata `--reads`는 정렬이 실제로 쓴 파일(제외 사본 경로)이다.
8. **Athaliana FLIC 검증3런은 별도 드라이버로 정렬.** 기존 production 스크립트의 역할 검사는 바꾸지 않는다. §3.3 방향성 감사와 primary 정렬을 같은 도구·명령으로 수행하는 별도 스크립트를 쓴다. 검증 전용이라 audit 정렬(-N50)은 필요 없다(`b6_build_c0.py:685`).

## 9–11. HSP 계약·A16 linkage 결정 (2026-09-26)

사용자 원문: `(a) 연결된 묶음 전체를 한 분자로 합치기 (추천). 1. HSP 길이 계약: 행마다 M+I+D나 M+I+D+F 중 하나와 맞으면 통과시키고, 어느 쪽인지 QC에 기록합2. 실제 서열로 재생되지 않는 HSP 행: 잡을 멈추지 말고 그 행만 빼고, 이유별 개수를 QC에`

9. **A16 비추이적 10-nt PCR 동치 = (a) 연결성분 전체를 한 분자로 합친다.** UMI absent cDNA ONT의 (library, contig, strand, intron chain) 묶음에서 양 끝 10nt 동치 그래프의 연결성분 하나가 분자 하나다. clique가 아닌 성분도 거부하지 않는다. 비-clique 성분 수와 read 수를 QC에 기록한다.
10. **HSP 길이 계약:** 행마다 DIAMOND 보고 길이가 M+I+D 또는 M+I+D+F 중 하나와 정확히 같으면 통과. 어느 쪽으로 통과했는지 QC에 기록한다. 둘 다 아니면 11의 제외 대상.
11. **실제 서열로 재생되지 않는 HSP 행:** 잡을 중단하지 않고 그 행만 제외하며, 제외 이유(메시지 종류)별 개수를 QC에 남긴다. 제외는 fail-closed(투영에 쓰지 않음)이다.

## 12. Bdistachyon exhaustiveness: 감사 N=100 재실행 (2026-09-26)

사용자 원문: `A AND n=100 IS GOOD. WATCHER 20 MIN NOT 5 MIN`

- Bdistachyon(EST 단독, 종 exhaustiveness 0.99897 < 0.999)의 감사 정렬을 **N=100**으로 다시 돌리고 EST C0를 `--audit-n 100`으로 재빌드한다. 종 내 모든 소스는 같은 N을 쓴다(Bdistachyon은 EST뿐).
- Gmax·Vvinifera가 ONT 결과 후에도 0.999 미만이면 같은 규칙(N=100, 종 내 전 소스)을 적용한다.
- 감시 주기는 20분.

## 13. Gmax·Vvinifera도 N=100 즉시 적용 (2026-09-26)

사용자 원문: `apply the same N=100 rule to all of that species' sources`

- ONT 결과를 기다리지 않고 Gmax(EST+ONT)·Vvinifera(EST+PacBio+ONT)의 모든 소스를 감사 N=100으로 다시 정렬하고 C0를 재빌드한다. 해당 종의 N=50 C0는 병합에 쓰지 않는다.
