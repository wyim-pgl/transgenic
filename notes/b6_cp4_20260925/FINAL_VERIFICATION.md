# B6 GPU 구축 검증 완료: 2026-09-25 01:03 PDT

> 후속 갱신(09-25 12:55PDT): 사용자 승인으로 NVMe 정리도 완료했다. 146.262GiB를 확보했으며, 과거 B6 4트리는 HDD archive로 전수 검증 보관했다. 아래 ‘정리 승인 대기’는 이 문서 작성 시점의 상태다. [정리 정본](../b6_nvme_cleanup_20260925/REPORT.md).

## 완료 범위

| 검사 | 잡 | 원본 검증 결과 |
|---|---|---|
| 129,024 nt 메모리 벤치 | 3208285 | PASS, GPU2, rank당 peak reserved62.959GB≤100GB, Inductor/비재진입checkpoint, optimize_ddp=false |
| 정식 CP4 WP3 fixtures | 3210279 | PASS, 실제NCCL fault7시험PASS, 재개step3바이트일치, 양rank greedy/beam 시퀀스일치, 3link 입력결속동일 |
| 정식 CP4 synthetic | 3210585 | PASS, 실제NCCL fault7시험PASS, 재개step5바이트일치, 양rank greedy/beam 시퀀스일치 |

세 잡 모두 sacct COMPLETED0:0. 각각 실행3분17초, 6분31초, 7분22초. 01:03PDT 확인 시 활성B6잡 없음.

각 CP4의 straight/resumed run_config에서 world_size2, bf16, nonreentrant 및 `torch.compile backend=inductor; suppress_errors=False; optimize_ddp=False; fallback=failure`를 직접 확인했다. 메모리벤치와 현재 모델·bench·config바이트 및 compile_b6 AST 연결도 검증했다. 전체 source/config/test220파일을 실행 후 다시 대조해 모두OK다.

## 체크포인트와 보고서 해시

- WP3 straight/resumed step3: 둘다 `05a396a95e5faac8f77ae9a41e86e16d19de6d38a2c0bd4d37969a14499270d0`.
- synthetic straight/resumed step5: 둘다 `6b79d0398551faacf5a21d8f58ce8f41186c2f1ca39eccb4ccbfdc533a55eeb0`.
- 메모리 report JSON: `1f66b93a7d77b41cc69501082e544a102917f26f14663e0d06e135ea583f6211`.
- WP3 report JSON: `62da85b2efc8d01a41bdb737523fcdce49d75987a5f833684f1bf844fd3e92e4`.
- synthetic report JSON: `2f59a73b3c3e59a0a679f155c1d8a6a537d83188bf98d10a873ac99943c56118`.

원본 대조 로그: `wp3_verified.txt`, `synthetic_verified.txt`, `gate_linkage.txt`, `final_live.txt`.

## 코드 정본

- 로컬 `/tmp/b6-integration`, 브랜치b6-integration, HEAD **3511a5a3fe2eec1d3ef318354adbed060de12a9f**.
- 수정 커밋: c9e242c(캐시API), a48579d(CUDA배치검증), f1ee718(unsplitcompile/정책결속), b4dd15e(checkpoint오류전파/quota/HDD), 3511a5a(시험FileStore).
- 원격 `/work/hdd/bilv/wcyim/transgenic/repo_b6_filestore_20260924`, 스냅숏e2a91b7f4485e5eeb3b23d40fee0cac9a8a6bc10.
- main 병합·푸시 없음. B5 코드/잡 및 이전 실패 산출물 보존.

## 해석과 남은 차단

이 결과는 A51의 **2-GPU 기술 검증**이며 4-rank 실행을 측정했다는 뜻이 아니다. 과학적 결과 승인·운영학습 승인·C0 실데이터 구축 완료를 뜻하지 않는다. fixture/synthetic 체크포인트를 운영학습 초기값으로 사용하지 않는다. 동결 config의 gate_status도 임의로 수정하지 않았다.

요청받은 메모리 벤치와 두 CP4 구간은 완료됐다. 새 GPU잡 제출은 없다. 남은 현재 범위 항목은 **NVMe147GiB 정리 승인**으로, 기존 파일을 삭제하거나 이동하지 않았다. NVMe quota 여유 자체는 아직 복구되지 않았다.

Advisor 도구는 버전 오류로 검토 불가 상태이며, Advisor가 최종 승인했다는 주장은 하지 않는다. 코드 수정마다 독립 Codex 검토와 실행 검증을 수행했다.

30분 확인 예약4GzTTiZ04g는 변경하지 않았다. 확인할 새 B6잡은 현재 없다. 기존 terminal결과를 반복 해시하거나 Advisor에 재요청하지 않고, 새로운 지시/잡이 생기기 전에는 완료 상태만 유지한다.
