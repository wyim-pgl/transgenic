# Controller接口/파일소유권 — 2026-09-17

## 검토한 조사 결과
세readonlyPlan담당결과를controller가확인했다. Plan도구는write도구가없어보고파일을작성하지않았으므로이문서를실행요약정본으로남긴다. 원문은각agent출력에존재.

### 1. Seed
기존CLI choices1/2/3,readiness seedbinding,real_probes.setup,finalizermetadata seedallowlist,resourcebudget의3seed분모가실제변경대상이다. Recipe250updates/12warmup/4GPU×24accum96은유지.
- 새 `--profile development-single --seed123`; profile이없는과거기록은historical-three-seed만허용.
- 공유 `transgenic.training.mvp_profiles.validate_seed_profile(profile,seed)`; 정확int검증(bool/float/string거부).
- 최종metadata.profile/run_role=mvp_finetune,seed123/aug20261037; identity_scopeproduction은실hardware검증을뜻하므로test로낮추지않음.
- schema1legacy3seed예산보존, schema2development-single/[123]/max_training_node_hours는별도명시·실측검증. 새GPU예산임의생성없음.

### 2. Evaluation
- 개발requestprofiledevelopment-single, run_id=mvp_seed123(rolemvp_finetune)와b5_seed123(roleb5_baseline)을별도키로사용. seed숫자가같다는이유로모델/weights를혼동하지않음.
- 예측150=50off+50on+50baselineoff. checkpoint identity+role+fraction+locus로검증;historical350은보존/현재실행안함.
- 개발단일시드결과decision=null,3seedfinal인증없음;승인된sourceOR·missingness계약대로관측지표생성.
- oldpanel6165628manifest SHA7528c9...acc0e와150artifacts불변. 명시pinned externalcompatreceipt를추가하되원본producerinputs/소스경로/패키지기록검증,현재data/authority일치와seed123fullkeep1재래스터일치를실제검증한다. 기존currentproducerexact입력분기보존;임의oldsource면제없음. 새receipt만게시,패널/DB재생성없음.

### 3. Actual EST
Sitalica primary100을작은실자료시작점으로선정.16QNAME/양raw13전체소비로선택query의HSPscope확인→screened/original/report/filteredprotein/전체BAM선택record결속→실DNA/protein replay와geometry. Metadata없는A11은unassessable로유지,가짜분자ID/완결성불리언없음. Selfoverlap/noncanonical도원row와disposition보존,0support로해석금지.
Controller11:22readonlyfixedpath관측: query12.5MB/original15MB/report1.86MB/BAM18.7MB; orthoFA5.55GB/raw167MB, swissFA21MB/raw60MB. 작은provenance/commands/alignPROV를로컬에보존. 큰소스읽기/hash는compute에서만.
1차compute제안2CPU8GB30min,40GiBread/256MiBoutput/1Mbaseoccurrences 상한. 코드·정확manifest/runtime/wrapper독립검토전미제출. routeAminiprot실replay는후속연결,자료prefix만으로완료인증하지않음.

## 쓰기 담당 (동일source를공유해도파일은비중첩)

| owner | worktree | 소유파일 |
|---|---|---|
| 0f6198b9 seed/training | /tmp/mvp-seed123-integration-20260917 | training mvp_profiles/cli/runtime/finalize; readiness/real_probes/resource_probe/run_training;해당시험 |
| fae82185 eval | 같은worktree | mvp_decode_panel/produce_counts/decode_runtime/score, 필요bootstrap;해당시험 |
| 5d76780c panelcompat | 같은worktree | mvp_verify_panel_inputs + 새mvp_panel_compatibility,전용시험 |
| b0cb4695 actualEST | /tmp/est-real-source-binding-20260917(basefca911e) | 새est_real_sample.py/tests/tasknotes |

독립commit금지,controller통합시험/검토후commit. 공유helper는seed담당만작성,새panelhelperclosure는eval담당이runtime에결속. 실제sourcehash/P1release는seed+eval+panel통합이끝난뒤고정한다. 원격제출/취소담당은controller하나. 기존P1/b5/데이터수정없음.
