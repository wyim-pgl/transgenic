# Actual Slurm task-field correction

Prior release_v5 passed its scoped review, but controller's final comparison to retained real scontrol output found `CPUs/Task=64` while internal allocation checks expected CPUsPerTask. The raw parser preserved the former unchanged, so a genuine allocation would be rejected. No GPU transfer/submission occurred before discovery.

New implementation semantic change is exactly one parser line: map only `CPUs/Task` to internal `CPUsPerTask` before duplicate-key checking. Both aliases in one response reject, even if values agree. Wrong/missing values remain errors; no CPU/request/node-capacity limit is changed. All other implementation changes are newv6 namespace/authorization/job/build identities, leaving frozen prior releases unchanged.

Regression uses exact captured job3163986line SHA95d819c95d7665cb830f177c75f62db80585245d77dda26ac48159c80ad26679. It is explicitly historical PENDING evidence, not current state. Tests cover observed alias, both-alias ambiguity, entire synthetic admission using real-format spelling, changed final-snapshot taskCPU rejection, and no broad alias/default mapping.

Implementer completedRED/GREEN and release_v6directory before turnlimit. Controller completed exclusive transport creation, full47testsPASS11.273s, ledgerverification, exact14transportmember bytecomparison and305priorfile preservation checks. No partial work is silently reported as complete.

Pins:
- release_v6.tar26c1dc0a233a95ab86977813e68310eccfce91b081de3a63586cfd7b9427628a
- release.json64a547b8ba5f54e87c1985f278564a9ea9fc7c8f3af07f6b6afe3bdf17f4f7d4
- release.sha25674a740ea154532aa3c0b807d052fb9c05bdf02fcfb6fd86845d348be72f70d9b
- sourcecandidatearchiveunchangedda161088fb401dbe87382685c1b5faadc17c7255d4a89f705fe1126ecc5bcec7

Narrow independent review and controller authorization remain required. All prior fixed-failure-recovery, durableauthority/freshcomputechecks,5nodeh/20GPUhcap,0.5nodeh/2GPUhnewattempt,B5preservation,readonlyknowninputcopy and no-training conditions remain. No cancellation,transfer,job or sourceworktree edit performed.
