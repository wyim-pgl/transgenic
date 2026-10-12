# Verified deployment environment — 2026-10-09

## DeltaAI

- Access: local `ssh pronghorn`, then `ssh -o BatchMode=yes dtai`; resolves gh-login03.delta.ncsa.illinois.edu in this session.
- Host Python: `/usr/bin/python3`, 3.6.15. rsync and lfs available. No GPU needed for stdlib/small-file tests.
- NVMe runs root `/work/nvme/bilv/wcyim/transgenic/runs`; pool ddn_ssd, project22169, quota500GiB soft550GiB hard; free105.0907GiB at13:39CDT.
- HDD parent `/work/hdd/bilv/wcyim/transgenic`; pool ddn_hdd, project22169; quota1000GiB soft1100GiB hard; used829.286GiB, free170.714GiB at13:43CDT.
- Main HDD consumer `/work/hdd/bilv/wcyim/transgenic/b6_prep_20260925/merged/b6_v1/evidence_c0.db` =786283638784bytes (~732.28GiB). Do not modify input evidence.
- Both pools are one Lustre filesystem, st_dev3290199710. Rename keeps SSD pool; a fresh file under HDD inherits HDD. Actual 1MiB copy hash equality proved previously.
- fcntl.flock tested on each pool with a competing Python process: exclusive contender rejected; locks usable. /work mount options include flock (not localflock): rw,nosuid,nodev,flock,lazystatfs,encrypt. Cross-node contention itself has not been directly tested. Test files removed.
- All B5 and WP5 training links were pending at13:39CDT. User systemd is running, but no service/timer should be enabled for production cleanup implicitly.

## Pronghorn

- User wyim, archive directory group p-wyim.
- Python3 not on noninteractive PATH. Use `/data/gpfs/assoc/pgl/data/Transgenic/.venv-ssh/bin/python` = Python3.11.14.
- `/usr/bin/rsync` 3.1.2 on pronghorn; DeltaAI rsync3.2.7. Both support protected arguments. `ssh dtai` authenticated from pronghorn; implement PULL here.
- Destination parent `/data/gpfs/assoc/pgl/data/Transgenic/archives` writable.
- GPFS fileset assoc.pgl: mmlsquota reports40833GiB used, hardlimit61440GiB, ~20607GiB before in-doubt adjustment; df shows21T available. User quota for wyim and group quota for p-wyim in assoc.pgl both say no limits. Use destination free-space checks and conservative configured reserve.
- Project directory may contain a very large tree; query targeted metadata/known files, not whole-tree repeated du.

## Login-node scope

Official DeltaAI data-management documentation permits small/modest scp/rsync transfers and recommends Globus for large transfers: https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/data-mgmt.html

Implement and repeat tiny fixture tests on login nodes. Do not treat a99GiB production migration plus full hashes as a tiny test or enable an unbounded destructive watcher as part of installation.
