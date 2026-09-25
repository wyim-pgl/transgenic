# ONT 실험 방식 분류 승인

기록 시각: 2026-09-25 14:44 PDT.
사용자 결정 원문: `실험 방식 으로 채택`.

## 승인 범위

`data/manifests/ONT_PROTOCOL.b6_v1.draft.tsv`의 학습 대상(c2_training_eligible)125런에 대해 현재 protocol 분류를 채택한다.

- cDNA: 83런
- direct_RNA: 42런
- 대상 파일 SHA256: `e21425e1eeaa60ac87648464b684f61fabe06c848ec8d0de4da448d54695b029`

이 기록은 A49.1에서 남겨 둔 protocol 열의 저자 검토를 완료한다. TSV 바이트와 기존 library 키는 변경하지 않는다. 제외/평가용 런의 학습 역할을 바꾸지 않는다.

## 이번 승인에 포함하지 않는 것

- UMI 열의125런 `absent` 일괄 확정. 이 값은 공개 메타데이터의 무언급에 근거한 초안이며, 추가 근거 확인이 남아 있다.
- 다른 metadata 컬럼, 임의 library 생성, 입력 제외, 과학적 게이트 완화.
- 실데이터 C0/corpus/cache 준비 완료 또는 GPU 훈련 제출 완료라는 주장.

후속: protocol 승인과 UMI 판정을 분리해 입력 준비를 진행한다. UMI 추가 확인은 원자료/논문/키트 기록에 근거하고, 미확인 값을 확인된 미사용으로 바꾸지 않는다.
