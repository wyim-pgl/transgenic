# Quarantine — 현재 C2 복구에서 반복하지 않을 접근

현재상태는[resume](resume.md)/[handoff](../notes/RESTART_HANDOFF_20261012.md). 과거전체기록은[quarantine_history_20261012.md](quarantine_history_20261012.md)에보존했다. 실패방식과검증된대안을기록하며미완료실행은[todo](todo.md)에둔다.

| 반복하지 않을 접근 | 확인된 사실/대안 |
|---|---|
| legacyvalidator반환tuple을strictJSONcollective에그대로전달 | 3369338첫forward전실패. validator/replay분리시험외dispatch경계시험필요; 전송직전validatedpair만list화. |
| CUDA label구간경계를3414/4096포함여부확인없이기억 | 실제guard1536*labels<5242880는최대3413; 구간<=3413/3414..4095/>=4096. metadata검사는완전한compilerledger증명아님. |
| mark_dynamic annotation만signature에기록 | 실제maybe_mark_dynamic은weak_dynamic_indices사용. 789ecf4는strong/weak별도결속·26시험PASS. |
| parameter/buffer registry만복원 | 일반__dict__속성shadowing및sharedstorage/offset도복원필요. 91c2737·12시험·독립리뷰PASS. |
| userquota로NVMe가용량계산 | bilv는group이아님; project22169/poolddn_ssd softlimit500GiB기준. 10-12재개사용349.4GiB. |
| 3364480 Slurm0만으로복구완료/본학습승격 | 첫forward통제실험만통과. 운영복구코드는미구현,전체checkpoint연속/재개정확일치필요. |
| 입력draw/RNG일치면재개결정성도일치한다고판정 | 3361666은입력tensor·모든RNG동일인데609모델tensor/optimizer불일치. 실제가중치/gradient대조. |
| 최근3/18guard수정이손실차이를만들었다고가정 | 동일조건old/new200→210은모델/optimizer/RNG까지정확일치. 원인은compilerhistory로분리됨. |
| DDPbucket재구성을원인이라고추측 | telemetry has_rebuilt_buckets=0. primed/reset대조로컴파일상태의존성확인. |
| 첫3DNA tier만재생하면legacy200복원충분 | rank3row3915(labels4501)추가compile은200이전. 모든기존특수화이력/첫hint를결속하고증명. |
| rank별서로다른event목록을독립DDPreplay | collective순서불일치위험. 동기화된전역slot별실제각rank companion사용. 임의padding처방금지. |
| 5primer증명에서event-only압축을자동일반화 | 압축전후정확일치추가증명필요. 처음샘플hint가fusion/수치결과에영향. |
| torch._dynamo.reset이diskcache도초기화한다고가정 | 실제NGCsource는filesystemcache미변경. reset통제에서는FX/AOT local/remote lookup을모든case에서동일하게끔. 운영설정무단변경금지. |
| weights만되돌리고priming완료 | parameter객체·None/new/nonpersistent전체bufferregistry·모든RNG·pendinggrads·mode복원. optimizer/scheduler/step진행금지. |
| 과거continuationcontroller3276938재시작 | 중단journal blocked_determinism/jobs[]. 새수정과검증receipt로새controller/저널설계. |
| 3361386 FAILED를새trainer장애로판정 | 의도적USR1으로256정상저장·종료. batch1은400최소시험조건미달. |
| 오래된종료잡을squeue -j로활성판정 | 종료ID는Invalid job id일수있음. squeue -u로활성목록,sacct로종료결과확인. |
| 786GB입력SHA를login에서실행 | 무거운검증은Slurm잡내실행. login은작은fixture/조회만. |
| hostPython을컨테이너Python과동일취급 | DeltaAI host3.6.15,NGC3.12. gzip.GzipFile로감싼tar(modew|)사용; w|gz의compresslevel인자미지원. |
| run전체rglob후cache만필터해증거수집 | 캐시순회55초timeout재현. root/case별필요json/jsonl/log/meta만명시수집. |
| NVMe→HDD mv만하면pool이바뀐다고가정 | 같은Lustre에서mv는SSDpool유지. 새HDD복사→해시/내구성확인→원본정리. |
| 과거빈index.lock때문에git add실패 | 09-14생성0byte·활성git/열린FD없음확인후 .git/index.lock.stale-20261012로보존이동. 활성잠금은제거하지않는다. |
| /tmp worktree를영구정본으로취급 | 브랜치커밋이정본. 없어지면worktree재생성. main에는구현브랜치병합하지않음. |
| pronghorn기본tmux서버kill | protocol불일치때전용socket codex-archive-20261009사용. 기존서버보호. |

추가고정규칙: pronghorn s1-pgl/s2-core만사용(s3-sponsored금지). Slurm.extern OOM0:125는저자지시에따라판정에서제외하되batch OOM은실제실패다. 증거DB786GB·원본checkpoint200·B5latest/best/최근epoch는보호한다. 사용자10-12명시승인에따라수정·검증후C2본학습에승인을다시묻지않는다.
