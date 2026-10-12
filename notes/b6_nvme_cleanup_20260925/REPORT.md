# 승인된 NVMe B6 산출물 정리 완료

완료 확인: 2026-09-25 12:55 PDT. 사용자 지시: NVMe 정리. 기존 승인대기였던 B6 시험 산출물 범위만 처리했다.

## 결과

- NVMe pool 사용량: **548.266 GiB → 402.004 GiB**.
- 실측 확보: **146.262 GiB**. soft quota500GiB 아래로 복구, soft 기준 여유97.996GiB.
- HDD archive: `/work/hdd/bilv/wcyim/transgenic/archives/b6_nvme_20260925/payload/`.
- 414개 정규 파일, 전체528개 항목, 논리크기157,045,051,743 bytes를 보관했다.
- 모든 정규 파일 SHA256 및 디렉터리/파일 모드, 타입, 심볼릭 링크 대상이 원본과 일치했다. 범위 내부 하드링크 묶음도 확인했다.
- source/archive manifest SHA256 둘다 `56238a8f9546e549bd8db4a63c2ce0dc8c745882ee86a4e05406e6551f76b18f`.

## NVMe에서 제거한 정확한 범위

기준 `/work/nvme/bilv/wcyim/transgenic/runs/`:

1. `b6_int4/run_3207149`
2. `b6_int4/run_3207331`
3. `b6_int4/run_3208442`
4. `b6_resume_diagnostic/job_3207863`

위 상대 경로가 HDD archive payload 안에 그대로 있다. NVMe의 두 부모 디렉터리는 빈 상태로 남겼다. 다른 run, B5 데이터·잡·코드, source repo, 메모리 벤치 결과, HDD의 최신 PASS 결과는 삭제하지 않았다.

## 안전 검증

- 복사 전과 삭제 직전에 B6 활성/대기 잡 부재 및 해당4잡 terminal 상태를 확인했다.
- 새 archive 디렉터리만 생성, 기존 archive 덮어쓰기 없음. source 전체 해시 → rsync -aH → archive 전체 해시를 비교했다.
- 원본·사본의 inode/dev/size/mtime/ctime/link count 스냅숏으로 복사·검증 뒤 변경이 없음을 확인했다.
- HDD 파일·디렉터리·부모를 fsync하고 삭제시도 저널을 먼저 영속화했다.
- 각 승인 root의 inode 등 속성을 재확인하고 private0700 staging으로 atomic rename 후 다시 같은 객체인지 확인한 뒤 삭제했다. symlink-safe rmtree만 사용했다.
- 독립검토 초기 지적(root 교체 경쟁·하드링크 검증)을 보강한 후 재검토 `3d57ee5c-3116-42d` APPROVE.
- 로컬 실제 임시파일 시험: 정상검증삭제, 사본변조시 원본보존, 하드링크 topology불일치시 원본보존 모두 통과.
- 삭제 후 네 NVMe root 부재, HDD archive 전체 스냅숏 불변, 메모리3208285 보고서 및 HDD PASS3210279/3210585 보고서 보존을 확인했다.

## 증거

- `archive_cleanup.py`: 실행한 두 단계 도구.
- `copy-verify.log`, `verification_receipt.txt`, `delete-verified.log`, `after.txt`.
- 원격 archive 루트의 `source_manifest.json`, `archive_manifest.json`, `source_stamps.json`, `archive_stamps.json`, `VERIFIED.json`, `DELETE_STARTED.json`, `DELETED.json`.

GPU 잡 신규 제출·운영학습·main 병합·푸시 없음. 승인된 현재 정리 TODO는 완료했다.
