# Controller scoped ONT F1-F3 check

Controller read FIX_REPORT.md, inspected the complete backup-to-current production diff, and independently ran `/tmp/mvp-final-source-env/bin/python3.12 -I -B ont_full/test_wrapper.py -v` with a90second outer deadline. Result14testsPASS in14.641s; log ont_full/controller_fix_tests.txt.

The production changes add exact nine-file/required-directory inventory checks before gate import and after execution, reject unsupported Python before path access, and compare both original streams plus final diagnostic identities before publication. No old gate/source pin was changed. This addresses the specified local F1-F3 behavior; exact immutable deployment and externally approved Python installation remain trust prerequisites. Scoped controller check accepted, not production readiness.

F4 remains open. Additional controller lookup found minimap2 outside PATH in existing conda environments. `/home/wyim/miniconda3/envs/test/bin` contains minimap2 2.22-r1101, samtools1.12 with htslib1.16, seqkit2.3.0, but Python3.7.12. It is not the approved remote interpreter/runtime, and must not be silently substituted to satisfy F4. No environment was activated, modified or installed. A separately scoped real-tool fixture with approved layout/interpreter/tool versions, live Slurm/cgroup behavior, resource budget and extern accounting treatment is still needed before canonical ONT execution.

No actual alignment, scores, scientific availability or source substitution accepted. Missing evidence remains unassessable.
