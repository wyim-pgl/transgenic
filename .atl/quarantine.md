# Quarantine — 현재 C2 복구에서 반복하지 않을 접근

현재상태는[resume](resume.md)/[handoff](../notes/RESTART_HANDOFF_20261012.md). 과거전체기록은[quarantine_history_20261012.md](quarantine_history_20261012.md)에보존했다. 실패방식과검증된대안을기록하며미완료실행은[todo](todo.md)에둔다.

| 반복하지 않을 접근 | 확인된 사실/대안 |
|---|---|
| MVP50 평가기를 WP5라고 이름만 바꿔 사용 | source198ce14의mvp_decode_panel은50좌위·MVP모델·8채널계약을강제한다. WP5는pilot300·B6 31채널+chain·fixed4099·C2/C5가필요하다. [현재감사](../notes/ccloop_current_model_20261021/WP5_READINESS_AUDIT.md)를따라별도연결·검증해야한다. |
| MVPbootstrap의5/95분위수를WP5의95%구간으로무검토재사용 | mvp_bootstrap.py:161-189는MVP의단측경계를반환한다. 동결A46규칙13의구간의미와대조·독립검토하고WP5통계시험으로확인해야하며임계값을임의로바꾸지않는다. |
| 문서의 끊어진 상대 Markdown 링크를 저장소 루트 파일로 자동 연결 | Graphify 병합 검토에서15개 잘못된 root fallback을 재현했다. Markdown 링크는 포함 문서 기준으로만 해석하고 실제 존재하는 그래프 대상에 연결한다. |
| 단순 Graph 재출력으로 같은 노드 쌍의 다른 관계를 버리기 | Graphify 초기 병합 검토에서 다중 관계 손실 발견. 검증된 original_links를 초기·군집명 출력 모두에 보존, multigraph 표시 및 최종38,057관계/연결 대상0누락 검증. |
| semantic 인용 위치 null을 정밀한 행 인용으로 꾸미기 | 실제로 읽은 파일 전체 범위만 명시하고 location_precision=whole_source로 표시한다. 기존 정밀한 행·페이지 인용은 유지한다. |
| 동일하게 빈 optimizer/scheduler/RNG를 두 checkpoint에서 비교해 READY | e2bee04는 실제 prepared model/optimizer/scheduler schema8파일 및 fullAdamW/four-deviceRNG/기존slot 보존을 독립 검증한다. 단순두파일동일성만으로 완전한 상태를 증명하지 않는다. |
| scancel --state=PENDING --user만으로 원자적 필터를 주장 | 일반scancel은 client snapshot을필터한뒤jobID RPC를보낼수있다.2a48265는 `--ctld`를추가해controller-side state/user필터를사용한다. JobHeldUser reason의원자적보장은없다. |
| Apptainer --containall의 작은 /tmp에서 다수 fixture copy | NGC최초20setup ENOSPC 재현. 소유host/tmp scratch를컨테이너/tmp에명시bind 후36/36PASS,원래실패로그보존/시험scratch정리. |
| epoch09가 정리된 archive3425279 wrapper를 처음부터 재실행 | wrapper는09/10/11고정및원본존재를검증한다. 중단시각epoch manifest/verified/removed와partial을읽어별도복구범위를검토한다. |
| 큰stagingpayload를중첩SSH argv에삽입 | 로컬MAX_ARG_STRLEN(E2BIG)로원격실행전에실패. payload는stdin,작은실행코드만argv로전달한다. |
| 호스트 전용 시험 symlink를 운영 bundle 아래 남기기 | B5 3309786은 C2 host smoke의 절대 `/work/nvme/...` 링크가 컨테이너 `/work` bind 안에서 dangling이 되어 preflight 실패. 10-20 동일 이미지/bind로 재현 후 소유 시험 링크만 제거, 대상 receipt 보존 검증. 실제 입력 검사 게이트를 완화하지 않는다. |
| 완료 C2의 `latest_state=4000`을 최종4099 미달로 판정 | 본학습3379630의 최종은 별도 `resumed/step_04099`; meta/global4099, PILOT_FINAL, PRUNE_EXEMPT, TRAINING_DONE와 allocated postflight receipt를 확인한다. controller complete이므로 재시작/중복제출 금지. |
| compiledcheckpoint키를baregrad/export키와직접대조 | Accelerator저장은_orig_mod.*; gradient/fixedexport는bare. 7266e14는값digest유지·명칭대조만정규화, 실제compile/NGC시험·독립검토PASS. |
| CUDA RNG를nonemptyuint8만확인 | 1byte상태도READY가될수있음. 7266e14는sourceencode_rng+4기기Philox검증, NGCCPU33시험PASS. |
| localRENAME_NOREPLACE성공을실제Lustre지원으로일반화 | 실제host새디렉터리/파일rename 모두EINVAL22. 기존targetEEXIST만으론지원증명아님; fallback금지. controller6e5f0f7의in-place v2방식으로원본이동없이해결. |
| destination없음확인후os.rename면덮어쓰기없다고가정 | 경쟁빈디렉터리가생기면교체됨. atomic no-replace필요,미지원시중단. |
| helpermanifest만검증하면임의phase.script도검토된것 | 실제제출스크립트가해당helper내정확한manifestmember인지결속해야함. |
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
