"""Replay measured quota through the production preflight with mocked read-only OS probes.
Usage: python3 storage_repro.py /tmp/b6-compile-static
No remote files or quota values are changed.
"""
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

repo = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("storage_probe", repo / "revision/scripts/b6/storage_preflight.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
root = "/work/nvme/bilv/wcyim/transgenic/runs/b6_wp5_pilot_C2"
free = 112840318976
checkpoint = 5727678235
limit = 524288000
print("quota_remaining_GiB", free / 2**30, "required_GiB", m.MIN_FREE_BYTES / 2**30,
      "margin_GiB", (free - m.MIN_FREE_BYTES) / 2**30, "checkpoint_GiB", checkpoint / 2**30)
for label, extra in [("current_snapshot", 0), ("after_one_new_B6_checkpoint", checkpoint)]:
    used = limit - free // 1024 + math.ceil(extra / 1024)
    def command(argv, **kwargs):
        return {"project": f"22169 P {root}", "getstripe": "pool: ddn_ssd",
                "quota": f"/work/nvme/bilv {used} {limit} 576716800 - 98589 0 0 -"}[argv[1]]
    with patch.object(m.Path, "resolve", lambda self, strict=True: Path(root)), \
         patch.object(m.os, "statvfs", lambda _: SimpleNamespace(f_bavail=296210501632, f_frsize=1)), \
         patch.object(m.subprocess, "check_output", command):
        try:
            value = m.check_storage(root)
        except RuntimeError as error:
            assert extra and "insufficient CP4 storage" in str(error), str(error)
            print(label, "REFUSED", str(error))
        else:
            assert not extra, "Expected quota gate to reject the additional checkpoint"
            print(label, "PASS", value["quota_remaining_bytes"])
