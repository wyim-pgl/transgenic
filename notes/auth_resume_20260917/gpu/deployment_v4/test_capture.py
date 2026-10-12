"""Real harmless local children exercise capture bounds and cleanup."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import admission as a

class CaptureTests(unittest.TestCase):
    def test_non_timeout_read_failure_kills_and_reaps(self):
        children=[]; real_popen=subprocess.Popen; real_read=os.read
        def launch(*args,**kwargs):
            p=real_popen(*args,**kwargs); children.append(p); return p
        def broken_read(fd,size):
            if children and fd in (children[0].stdout.fileno(),children[0].stderr.fileno()):
                raise OSError('synthetic capture read failure')
            return real_read(fd,size)
        with tempfile.TemporaryDirectory() as tmp, patch.object(a.subprocess,'Popen',side_effect=launch), \
             patch.object(a.os,'read',side_effect=broken_read):
            try:
                with self.assertRaises(ValueError):
                    a.query(Path(tmp),'read-failure',[sys.executable,'-c',"import os,time;os.write(1,b'hello');time.sleep(1)"],timeout=3)
                record=json.loads((Path(tmp)/'read-failure.json').read_text())
                self.assertIn('synthetic capture read failure',record.get('error',''))
                self.assertIsNotNone(children[0].returncode)
                self.assertLess(children[0].returncode,0)
            finally:
                for p in children:
                    if p.poll() is None: os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=2)

    def test_aggregate_stdout_stderr_overflow_bounded_and_never_admitted(self):
        for fd in (1,2,0):
            with self.subTest(fd=fd), tempfile.TemporaryDirectory() as tmp:
                # At fd0, each stream is below the individual ceiling but their sum exceeds it.
                limit=16*1024*1024
                code="import os,time; data=b'x'*65536\n"
                if fd:
                    code+="for i in range(257): os.write(%d,data)\n" % fd
                else:
                    code+="for i in range(129): os.write(1,data);os.write(2,data)\n"
                code+="time.sleep(1)\n"
                with self.assertRaises(ValueError):
                    a.query(Path(tmp),'overflow',[sys.executable,'-c',code],timeout=4)
                record=json.loads((Path(tmp)/'overflow.json').read_text())
                self.assertIn('aggregate output',record.get('error',''))
                self.assertLessEqual(len(record['stdout'].encode())+len(record['stderr'].encode()),limit)
                self.assertLess(record['exit_code'],0)
                self.assertLess((Path(tmp)/'overflow.json').stat().st_size,limit+10000)

if __name__=='__main__': unittest.main()
