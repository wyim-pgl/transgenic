# A46 round-3 문안 지적 반영 기록

작성: 2026-09-14 15:25 PDT. 범위: 기존 리뷰의 남은 8개 문안/인터페이스 지적 반영과 관련 영문·한글 요약 동기화.
**문안 반영·로컬 검증 기록이지 최종 리뷰 승인, 저자 승인, 사전등록 동결 또는 실험 완료 기록이 아니다.**

## 검토 근거

- 기존 리뷰 원문: `/tmp/claude-1000/-mnt-f-scratch-develop-Transgenic/38e5c7a3-97dc-458a-ab85-62e458a459c6/scratchpad/codex_review_a46_r3_output.txt`
- 원문 SHA256: `87385460f82fc80cc53ff88aa5b748e15b7adb9347cb7f3c1d7dbd7e746f784e`.
- 원문의 판정은 **REJECT — eight wording/interface corrections remain**였다. 원문을 직접 읽고 아래 수정과 대조했다. 원문을 수정하거나 새 승인 판정으로 바꾸지 않았다.

## 지적별 반영

| # | 지적 | 반영 위치와 내용 |
|---|---|---|
| 1 | 승인 전 모든 미래 해시를 요구하는 모순 | `revision/protocols/PROTOCOL_B1_frozen_v1.md` A46 rule 15: rule 12의 승인 시점 명세·기존 산출물 해시와, 이후 생성 산출물의 최초 사용 전 해시 기록·검증을 구분 |
| 2 | B6/C2의 같은 예산 체크포인트 누락 | A46 rule 11: WP5 B6/C2 정확히 4,142 update, 최종 스텝 체크포인트로 C3/C4/C6의 같은 스텝과 비교; 본 런 best-validation과 구분 |
| 3 | ≥2 introns 문안과 실제 적격 풀 불일치 | `revision/protocols/B6_MVP_PILOT_v1.md` §1.2: 전사체별 최대 exon 수 ≥2, 즉 intron ≥1; A22/A30 직접 배제; A46 rule 17의 MVP/pilot/final 소속 문구 수정. 풀 재생성 없음 |
| 4 | 분자 계수 정의 3종 불일치 | MVP §1.2–1.3와 A46 rule 12: `(chromosome, FLAG-0x10 alignment strand, POS, CIGAR)`, POS 1-based; `ts:A`와 구분, clone/accession rule (b)는 C0-lite만. 기존 72/72a 설명은 이미 이 정의로 수정되어 있어 그대로 보존 |
| 5 | 패널 열·좌표·CLI 불일치 | MVP §1.2의 실제 14열, `start`/`fin` 1-based inclusive; §6의 실제 71/72a/72 인자 반영. 미구현 MVP 명령은 실행 가능한 명령과 구분해 sketch로 표시 |
| 6 | 개선량 부호·STOP/GO 동시 성립 | MVP §5.2–5.3 및 A46 rule 17: `Δ_loss = mean(loss_off − loss_on)`, STOP 우선·실패 시만 GO·그 외 INCONCLUSIVE, 이후 가용성에 따른 STOP 하향 판정. 증거 손실 부호와 F1/P/R 부호 구분 |
| 7 | 라이브러리 정보 부재를 근거로 한 부당한 withholding 면제 | MVP §5.4 및 판정문: 단일 소스/메타데이터 부재는 동등성 증명이 아님. withholding을 평가하지 못하는 prospective STOP은 INCONCLUSIVE |
| 8 | ONT 정렬 명령의 -k14/조건부 -uf 누락 | MVP §2: 프로토콜 §4의 ONT 두 명령과 HiFi 명령 원문 인용, A39로 교정된 ≥95% orientation audit에 의해서만 -uf 허용; UNRESOLVED는 미허용 |

A46 rule 17은 MVP §5.3의 원문과 Markdown 강조·줄바꿈을 제외하고 정확히 일치한다. MVP는 **v1.2 draft, 325 lines**다.

## 해시 및 보존

| 산출물 | SHA256 |
|---|---|
| MVP v1.2 draft | `69968c4d17d6362a8267f842f4e97735d7cfd6907eb3541751141b85fcf867e8` |
| 적격 TSV | `7ad895fcc25abf75c8fd682acd448c706c5d7f59c22da7c6aea0c8404219ecd6` |
| EST 계수 TSV | `36de5b6827cdba2b27c72a663cfb4b2a8c46a567a1d31a6450f438be604e869c` |
| 패널 TSV | `79367154bbf922fd0554285cb6cb47ae3d3c866b145aa1f51ead4f47b00cd243` |
| script 71 | `b81ec50baf88e10147c92bba390df5c63daf9fafc644d772f78df6080d86391c` |
| script 72 | `eab68edb76a7bb4397b1bef413fc47d68b4889553715e9efefc6573d9264f0d1` |
| helper 72a | `9e27472eb29798f8008275001293cfbf7dd23e67538127a5f35c146f6311b272` |

세 TSV·sidecar 및 세 스크립트는 이번 수정 전후 SHA256이 동일하다. 72/72a는 이전 세션의 설명 수정 때문에 A46에 남은 옛 해시와 달랐으며, A46의 참조 해시만 현재 파일에 맞췄다. A46 앞의 기존 프로토콜 본문은 이번 수정 전과 동일하다.

## 검증

- `python -m pytest -q revision/scripts/tests/test_b6_panels.py`: **8 passed**.
- `git diff --check`: 통과.
- 세 TSV 각각 2,079행, 패널 ID 중복 없음, 20 core + 30 MVP + 300 pilot + 1,729 final, intron ≥1, MVP 중 intron 1인 17개 확인.
- TSV sidecar·A46 참조 해시 및 스크립트 참조 해시 일치 확인.
- 실제 14열·1-based inclusive 길이 계산·MVP 외 EST 계수 빈칸 확인.
- 문서의 script 72 인자로 **임시 디렉터리에만** 재생한 패널이 기존 TSV와 바이트 동일. 원본 풀·패널 파일은 덮어쓰지 않음.
- MVP §5.3/A46 rule 17 인용문, MVP SHA256·행수 두 참조, §4 정렬 명령 세 개의 일치 확인.
- 판정 문안의 아래 11개 예시를 임시 Python 함수로 확인. **MVP scorer 구현 테스트가 아니라 문안 해석의 경계값 점검**이다.

| 예시 | 결과 |
|---|---|
| primary bounds [0.001, 0.004], 세 시드 양수, 안전조건 충족 | STOP |
| 위 조건에서 구조 채점 없음 | INCONCLUSIVE |
| 위 조건에서 day-5 정렬 미완료 | INCONCLUSIVE |
| 위 조건에서 withholding 평가 불가 | INCONCLUSIVE |
| 위 조건에서 withholding이 상한 ≥0.005로 반전 | STOP 실패 후 원래 primary GO 조건으로 GO |
| 위 조건에서 정의된 2차/구조 반증 있음 | primary GO 조건도 충족하므로 GO |
| primary bounds [-0.001, 0.004], 2차/구조 반증 있음 | INCONCLUSIVE; 반증만으로 GO가 되지 않음 |
| primary bounds [0.001, 0.005], 세 시드 양수 | GO; STOP 상한은 엄격한 <0.005 |
| primary bounds [0, 0.006] | INCONCLUSIVE; GO 하한은 엄격한 >0 |
| primary bounds [0.001, 0.006], 한 시드 비양수 | INCONCLUSIVE |
| primary bounds [-0.002, 0.003], 반증 없음·안전조건 충족 | STOP |

## 관련 문서와 남은 게이트

- 영문·한글 상세 워크플랜/README에 실제 50-locus MVP, 300/1,729 패널, 기존 저자 결정, C1' 생략·C2 마스크 동등성, G1만의 headline fallback, A46 승인 대기를 동기화했다.
- 기존 GPU 여유 약 50 h 계산은 MVP/C6 비용이 빠진 이전 추정으로 표시했다. 현재 잔액을 조회하거나 새 예산을 승인하지 않았다.
- 전체 실험의 과학적 타당성, 원격 BAM/DB/플래그 원본, 런 상태·런타임 정체성은 이번 문안 작업에서 재검증하지 않았다.
- 다음 단계는 수정본 최종 검토와 명시적 저자 승인이다. WP1 착수 금지는 유지한다.
- 커밋·푸시·원격 변경·학습/정렬 잡 제출 없음. 학습 코드·실험 데이터 변경 없음.
