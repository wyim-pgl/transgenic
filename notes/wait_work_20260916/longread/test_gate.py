import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
import gate

class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = 'SRR28341943'
        self.reads = self.root/'evidence/ont/Athaliana_FLIC_PRJNA1087576'/f'{self.run}.fa.gz'
        self.reads.parent.mkdir(parents=True)
        self.reads.write_bytes(b'fixture bytes, no alignment claim')
        self.roles = self.root/'evidence/DATASET_ROLES.longread_v1.tsv'
        self.row = dict.fromkeys(gate.ROLE_COLUMNS, '')
        self.row.update(dataset=gate.DATASET,run=self.run,species='Athaliana',genotype_stratum='reference',role='b1_validation_only',local_fa_md5=gate.digest(self.reads,'md5'))
        self.write_roles()
    def write_roles(self):
        with self.roles.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=gate.ROLE_COLUMNS,delimiter='\t');writer.writeheader();writer.writerow(self.row)
    def test_actual_read_binding(self):
        self.assertEqual(gate.read_binding(self.root,self.run,self.reads)['role'],self.row)
    def test_wrong_file_even_identical_bytes(self):
        other=self.root/'other.fa.gz';other.write_bytes(self.reads.read_bytes())
        with self.assertRaises(ValueError):gate.read_binding(self.root,self.run,other)
    def test_wrong_role(self):
        self.row['role']='c2_training_eligible';self.write_roles()
        with self.assertRaises(ValueError):gate.read_binding(self.root,self.run,self.reads)
    def test_wrong_md5(self):
        self.reads.write_bytes(b'mutated')
        with self.assertRaises(ValueError):gate.read_binding(self.root,self.run,self.reads)
    def test_wrong_reference(self):
        with self.assertRaises(ValueError):gate.reference_binding(b'wrong blob',self.reads,self.reads)
    def test_inventory_missing_or_changed(self):
        out=self.root/'out';out.mkdir();(out/'a').write_text('a')
        inventory=gate.inventory(out)
        gate.verify_inventory(out,inventory)
        (out/'unexpected').write_text('x')
        with self.assertRaises(ValueError):gate.verify_inventory(out,inventory)
        (out/'unexpected').unlink();(out/'a').write_text('b')
        with self.assertRaises(ValueError):gate.verify_inventory(out,inventory)
    def test_missing_done(self):
        with self.assertRaises((ValueError,FileNotFoundError)):gate.verify_orientation(self.root,'key',self.run,gate.load_driver())
    def test_real_frozen_orientation_scorer_fixture(self):
        source=gate.HERE/'frozen'
        gff=self.root/'fixture.gff3'
        gff.write_text('Chr1\tx\tgene\t100\t399\t.\t+\t.\tID=g\nChr1\tx\tmRNA\t100\t399\t.\t+\t.\tID=t;Parent=g\nChr1\tx\texon\t100\t199\t.\t+\t.\tParent=t\nChr1\tx\texon\t300\t399\t.\t+\t.\tParent=t\n')
        sam=self.root/'fixture.sam'
        sam.write_text('@SQ\tSN:Chr1\tLN:100000\n'+''.join(f'r{i}\t0\tChr1\t100\t60\t100M100N100M\t*\t0\t0\t*\t*\tts:A:+\n' for i in range(1000)))
        audit=self.root/'audit.json'
        proc=subprocess.run([sys.executable,str(source/'revision/scripts/67_orientation_audit.py'),'--sam',str(sam),'--gff',str(gff),'--out',str(audit),'--run',self.run,'--species','Athaliana','--dataset',gate.DATASET],capture_output=True,text=True)
        self.assertEqual(proc.returncode,0,proc.stderr)
        result=json.loads(audit.read_text())
        self.assertEqual((result['status'],result['valid_for_orientation'],result['read_sense'],result['use_uf']),('PASS',1000,1000,True))
    def test_orientation_identity_and_md5(self):
        driver=gate.load_driver()
        out=self.root/'orientation';out.mkdir()
        audit=dict(run=self.run,species='Athaliana',min_mapq=20,min_eligible=1000,threshold=.95,valid_for_orientation=1000,read_sense=950,status='PASS',use_uf=True)
        (out/'audit.json').write_text(json.dumps(audit))
        marker='key=a:b:c:d\nrun='+self.run+'\naudit_md5='+gate.digest(out/'audit.json','md5')+'\n'
        (out/'DONE').write_text(marker)
        self.assertEqual(gate.verify_orientation(out,'a:b:c:d',self.run,driver),audit)
        for bad in (marker.replace('a:b:c:d','wrong'),marker.replace(self.run,'SRR28341944'),marker+'key=a:b:c:d\n'):
            (out/'DONE').write_text(bad)
            with self.assertRaises(ValueError):gate.verify_orientation(out,'a:b:c:d',self.run,driver)
        (out/'DONE').write_text(marker)
        (out/'audit.json').write_text(json.dumps(dict(audit,read_sense=949)))
        with self.assertRaises(ValueError):gate.verify_orientation(out,'a:b:c:d',self.run,driver)
    def test_sample_never_full_coverage(self):
        for scope in ('orientation_sample','one_run','full_ont'):
            with self.subTest(scope=scope), self.assertRaises(ValueError):gate.require_final_coverage({'scope':scope})
    def test_failed_stdout_not_completion(self):
        out=self.root/'out';out.mkdir();(out/'stdout').write_text('PASS')
        with self.assertRaises((ValueError,FileNotFoundError)):gate.verify_publication(out)

if __name__=='__main__':unittest.main()
