# training_input — B5 학습에 들어가는 것 전부

ACCESS로 올릴 것을 한곳에 모은 스테이징 영역이다. **무엇이 들어갔는지는 `MANIFEST.tsv`가 정본**이고,
`bash make_manifest.sh`로 언제든 다시 만들 수 있다(파일별 bytes·md5·실경로·용도).

## 구성

| 디렉토리 | 내용 |
|---|---|
| `db/` | `b5_full_v1.db` — 동결 B5 학습 DB (272,224행; train 198,171 / valid 25,910 / test 48,143) |
| `splits/` | `b5_orthogroup_split_v1.tsv` — orthogroup 분할표. `random_split`를 대체한다 (#14) |
| `configs/` | 학습 레시피. **`b5_400m_win_v3.json`이 정본**(A26), `ctx_v2`·`v1`은 이력 |
| `manifests/` | 종 매니페스트 + **프로토콜 §1에 동결된 역할 매니페스트 2종** |
| `qc/` | GeenuFF·Swiss-Prot 플래그 18개 — A22 손실 마스킹 입력 |
| `protein/` | OrthoDB v12 Viridiplantae 누출 필터본 — C2 라벨 자원 (A19) |
| `freeze/` | B5 DB 동결 기록(내용 해시·provenance) |
| `genomes/` | **학습 9종만.** `Transgenic/genomes/`로의 심볼릭 링크이며, 표에는 실경로와 md5가 들어간다 |

## ⚠️ 게이트: 테스트 종 게놈은 못 들어온다

`make_manifest.sh`는 `genomes/`의 모든 파일을 `b5_species_v1.tsv`의 9종과 대조하고, 하나라도 맞지 않으면
**표를 쓰지 않고 종료한다**.

첫 적재에서 실제로 4개가 섞여 들어왔다 — `Zmays`·`Slycopersicum`(**held-out 테스트 종**)과
`BrapaO`·`Lsativa`. `Transgenic/genomes/`에 13개가 함께 있어 와일드카드가 전부 가져온 것이다.
`training_input`이라는 이름의 폴더에 테스트 게놈이 들어 있으면 배송될 때까지 아무도 눈치채지 못한다.

## 대조할 값 (프로토콜 §1 고정 객체)

| 파일 | md5 |
|---|---|
| `DATASET_ROLES.est_v1.tsv` | `a6375713ca2321c6a0cc55acc8e1d7fe` |
| `DATASET_ROLES.longread_v1.tsv` | `c2f4c2e8fe68b3e522abb8298e1999ff` |
| `odb12_Viridiplantae.filtered.fa.gz` | `453cb32b02e0799950d7d5f4de5f62ac` |
| `Athaliana_167_TAIR10.fa` | `513e5ef30845ed754b00816a99abbf8e` |

`MANIFEST.tsv`의 md5가 위와 다르면 스테이징이 잘못된 것이다.

## 여기 없는 것

- **장독·EST 정렬 BAM** — EST는 `evidence/est_align/`에 있고, 장독은 **아직 정렬 자체를 안 했다**
  (§3.3 정렬 전 감사 #25가 선행). 둘 다 B5 시드 학습 입력이 아니라 B1 증거 트랙이다.
- **miniprot 단백질 정렬** — #45 미실행. C2 트랙(#33·#34) 입력이며, `b5_400m_win_v3.json`이
  `"do_segment": false`이므로 **B5 시드 학습에는 필요 없다**.
