"""Offline CPU integration: real DuckDB and rasterizer; only tokenizer substituted."""
import multiprocessing
import pickle
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import duckdb
import numpy as np
import torch
from transgenic.datasets.datasets import isoformDataHyena, hyena_collate_fn


class Tokenizer:
    pad_token_id = 0
    def __call__(self, sequence, **kwargs):
        return {'input_ids': torch.tensor([[{'A':1,'C':2,'G':3,'T':4,'N':5}[b.upper()] for b in sequence] + [9]])}


def spawned_sample(ds, queue):
    queue.put((ds[0][-1], ds[0][8].tolist()))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = str(Path(self.tmp.name)/'corpus.db')
        self.ev = str(Path(self.tmp.name)/'evidence.db')
        with duckdb.connect(self.db) as c:
            c.execute('CREATE TABLE geneList(geneModel VARCHAR,start BIGINT,fin BIGINT,strand VARCHAR,chromosome VARCHAR,sequence VARCHAR,gff VARCHAR,static_fpb INTEGER,static_tpb INTEGER,five_prime_buf INTEGER,three_prime_buf INTEGER,rn BIGINT,species_id VARCHAR,split VARCHAR,train_weight DOUBLE,is_rc BOOLEAN)')
            for rn,sp,seq,rc,name,split in [(7,'Athaliana','ACNAAA',False,'keep','train'),(9,'Athaliana','TTTNGT',True,'skip','train'),(11,'Other','AAAA',False,'other','train'),(13,'Athaliana','AAAA',False,'valid','valid')]:
                c.execute('INSERT INTO geneList VALUES (?,100,?,?,\'Chr1\',?,\'\',0,0,0,0,?,?,?,1,?)',[name,100+len(seq),'-',seq,rn,sp,split,rc])
        with duckdb.connect(self.ev) as c:
            c.execute('CREATE TABLE corpus_tile(rn BIGINT,chrom VARCHAR,start0 BIGINT,end0 BIGINT,split VARCHAR,is_rc BOOLEAN)')
            c.execute("INSERT INTO corpus_tile VALUES (7,'Chr1',100,106,'train',false),(9,'Chr1',100,106,'train',true),(13,'Chr1',100,104,'valid',false)")
            c.execute('CREATE TABLE est_molecule(molecule_id VARCHAR,strand VARCHAR,excluded BOOLEAN,cause VARCHAR,primary_count BIGINT)')
            c.execute("INSERT INTO est_molecule VALUES ('m','-',false,NULL,1)")
            c.execute('CREATE TABLE est_block(molecule_id VARCHAR,chrom VARCHAR,start0 BIGINT,end0 BIGINT)')
            c.execute("INSERT INTO est_block VALUES ('m','Chr1',100,102),('m','Chr1',104,106)")
            c.execute('CREATE TABLE est_junction(molecule_id VARCHAR,chrom VARCHAR,donor0 BIGINT,acceptor0 BIGINT)')
            c.execute("INSERT INTO est_junction VALUES ('m','Chr1',103,102)")
            c.execute('CREATE TABLE tile_est(rn BIGINT,molecule_id VARCHAR)')
            c.execute("INSERT INTO tile_est VALUES (7,'m'),(9,'m')")
        p = patch('transgenic.datasets.datasets.AutoTokenizer.from_pretrained', return_value=Tokenizer())
        p.start(); self.addCleanup(p.stop)
        self.datasets = []
        self.addCleanup(lambda: [d.__del__() for d in self.datasets])

    def dataset(self, evidence=True, **kwargs):
        args = dict(species='Athaliana', evidence_db=self.ev, aug_seed=123) if evidence else {}
        args.update(kwargs)
        d = isoformDataHyena(self.db, **args)
        self.datasets.append(d)
        return d

    def test_legacy_and_species_branches(self):
        state = torch.get_rng_state()
        d = self.dataset(False)
        self.assertEqual(d._index_map, [7,9,11,13])
        s = d[0]; self.assertEqual(len(s), 8)
        b = hyena_collate_fn([s,d[2]])
        self.assertEqual(len(b),8)
        self.assertEqual(b[0].tolist(), [[1,2,5,1,1,1],[0,0,1,1,1,1]])
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        for kwargs, expected in [({},[7,9,13]),({'exclude_prefix':'skip'},[7,13]),({'split':'train','exclude_prefix':'keep'},[7,9])]:
            x = self.dataset(False,species='Athaliana',mode='train',**kwargs)
            self.assertEqual(x._index_map,expected)
            self.assertEqual(x.get_sample_weights(),[1.]*len(expected))
        self.assertEqual(self.dataset(False,species="Athaliana' OR 1=1 --")._index_map,[])
        self.assertEqual(len(hyena_collate_fn([s[:7]])),8)

    def test_tracks_orientation_mask_and_metadata(self):
        d = self.dataset(eval_fraction=1,return_tile_ids=True)
        a,b = d[0],d[1]
        self.assertEqual(len(a),10); self.assertEqual(a[-1],{'rn':7,'has_evidence':True})
        self.assertEqual(a[8].dtype,torch.float16)
        one = np.float16(np.log(2))
        self.assertEqual(a[8][3,5],one); self.assertEqual(b[8][2,3],one)
        self.assertEqual(a[8][2].count_nonzero(),0); self.assertEqual(b[8][3].count_nonzero(),0)
        self.assertEqual(b[0].tolist(),[[4,4,4,5,3,4]])
        self.assertIsNot(d._con,d._evidence_con)
        self.assertFalse(d[2][-1]['has_evidence'])
        self.assertEqual(len(self.dataset(eval_fraction=1)[0]),9)

    def test_collate_dna_not_label_length_and_validation(self):
        d = self.dataset(eval_fraction=1,return_tile_ids=True)
        a,b = list(d[0]),list(d[2])
        a[2],b[2] = torch.tensor([[1,2]]),torch.tensor([[3]])
        out = hyena_collate_fn([tuple(a),tuple(b)])
        self.assertEqual(out[2].tolist(),[[1,2],[3,-100]])
        self.assertEqual(out[8].shape,(2,6,8)); self.assertEqual(out[8][1,:2].count_nonzero(),0)
        self.assertEqual(out[9]['rn'].dtype,torch.int64); self.assertEqual(out[9]['has_evidence'].dtype,torch.bool)
        self.assertEqual(out[9]['rn'].tolist(),[7,13])
        for bad in [a[:8],a[:7]]:
            with self.assertRaises(ValueError): hyena_collate_fn([tuple(a),tuple(bad)])
        for tracks in [torch.zeros(5,8,dtype=torch.float16),torch.zeros(6,7,dtype=torch.float16),torch.zeros(6,8)]:
            bad = a.copy(); bad[8] = tracks
            with self.assertRaises(ValueError): hyena_collate_fn([tuple(bad)])

    def test_validation(self):
        for kw in [dict(species=None),dict(species='Other'),dict(evidence_k=7),dict(aug_seed=None),dict(aug_seed=True),dict(channel_dropout_p=float('nan')),dict(channel_dropout_p=2),dict(thin_fractions=()),dict(thin_fractions=(float('inf'),)),dict(eval_fraction=-1)]:
            with self.subTest(kw=kw),self.assertRaises(ValueError): self.dataset(**kw)
        with self.assertRaises(ValueError): self.dataset(False,return_tile_ids=True)
        isoformDataHyena.__new__(isoformDataHyena).__del__()

    def test_fail_closed_metadata(self):
        for sql in ["UPDATE corpus_tile SET chrom='bad'",'UPDATE corpus_tile SET start0=99','UPDATE corpus_tile SET end0=107',"UPDATE corpus_tile SET split='valid'",'UPDATE corpus_tile SET is_rc=true','DELETE FROM corpus_tile','INSERT INTO corpus_tile SELECT * FROM corpus_tile']:
            with self.subTest(sql=sql):
                with duckdb.connect(self.ev) as c:
                    saved = c.execute('SELECT * FROM corpus_tile').fetchall()
                    c.execute(sql)
                d = self.dataset(eval_fraction=1)
                with self.assertRaises(ValueError): d[0]
                d.__del__()
                with duckdb.connect(self.ev) as c:
                    c.execute('DELETE FROM corpus_tile'); c.executemany('INSERT INTO corpus_tile VALUES (?,?,?,?,?,?)',saved)

    def test_bad_corpus_and_token_lengths(self):
        for sql in ['UPDATE geneList SET is_rc=NULL WHERE rn=7',"UPDATE geneList SET sequence='' WHERE rn=7",'DELETE FROM geneList WHERE rn=7']:
            d = self.dataset(eval_fraction=1)
            with duckdb.connect(self.db) as c:
                saved = c.execute('SELECT * FROM geneList WHERE rn=7').fetchone(); c.execute(sql)
            with self.assertRaises((ValueError,RuntimeError)): d[0]
            d.__del__()
            with duckdb.connect(self.db) as c:
                c.execute('DELETE FROM geneList WHERE rn=7'); c.execute('INSERT INTO geneList VALUES ('+','.join(['?']*16)+')',saved)
        d = self.dataset(eval_fraction=1)
        for length in (0,4,8):
            with patch.object(Tokenizer, '__call__', return_value={'input_ids':torch.ones(1,length,dtype=torch.long)}):
                with self.assertRaises((ValueError,RuntimeError)): d[0]

    def test_missing_or_nonboolean_orientation_and_all_n_metadata(self):
        for sql in ['ALTER TABLE geneList DROP COLUMN is_rc', 'ALTER TABLE geneList ALTER COLUMN is_rc TYPE INTEGER']:
            with self.subTest(sql=sql):
                with duckdb.connect(self.db) as c:
                    c.execute(sql)
                d = self.dataset(eval_fraction=1)
                with self.assertRaises((ValueError,duckdb.Error)): d[0]
                d.__del__()
                with duckdb.connect(self.db) as c:
                    if 'DROP' not in sql: c.execute('ALTER TABLE geneList DROP COLUMN is_rc')
                    c.execute('ALTER TABLE geneList ADD COLUMN is_rc BOOLEAN DEFAULT false')
        with duckdb.connect(self.db) as c:
            c.execute("UPDATE geneList SET sequence='NNNNNN' WHERE rn=7")
        d = self.dataset(eval_fraction=1,return_tile_ids=True)
        self.assertFalse(d[0][8].any()); self.assertTrue(d[0][9]['has_evidence'])

    def test_private_rng_dropout_eval_and_thinning(self):
        state = torch.get_rng_state(); py = random.getstate(); npstate = np.random.get_state()
        a,b = self.dataset(),self.dataset()
        self.assertTrue(all(torch.equal(a[0][8],b[0][8]) for _ in range(20)))
        self.assertTrue(torch.equal(state,torch.get_rng_state())); self.assertEqual(py,random.getstate())
        np.testing.assert_equal(npstate,np.random.get_state())
        for p in (0,1):
            d = self.dataset(channel_dropout_p=p,thin_fractions=(1,),return_tile_ids=True)
            self.assertEqual(bool(d[0][8].any()),p==0); self.assertTrue(d[0][-1]['has_evidence'])
        e = self.dataset(mode='train',channel_dropout_p=1,eval_fraction=1)
        self.assertIsNotNone(e[0][2]); self.assertTrue(e[0][8].any()); self.assertIsNone(e._aug_generator)
        self.assertFalse(self.dataset(eval_fraction=0)[0][8].any())
        before = a._aug_generator.get_state().clone()
        a.eval_fraction = 1; a[0]
        self.assertTrue(torch.equal(before,a._aug_generator.get_state()))
        ts = [self.dataset(eval_fraction=f)[0][8][:,:7] for f in (.01,.1,.5,1)]
        for x,y in zip(ts,ts[1:]): self.assertTrue(torch.all(x<=y))

    def test_pickle_spawn_and_pid_worker_lifecycle(self):
        d = self.dataset(return_tile_ids=True); d[0]
        con,ev,g = d._con,d._evidence_con,d._aug_generator
        clone = pickle.loads(pickle.dumps(d)); self.datasets.append(clone)
        self.assertIs(d._con,con); self.assertIs(d._evidence_con,ev); self.assertIs(d._aug_generator,g)
        self.assertIsNone(clone._con); self.assertIsNone(clone._evidence_con); self.assertIsNone(clone._aug_generator)
        self.assertFalse(clone._validated_tiles)
        d._worker_pid = d._evidence_pid = -1
        d._aug_key = None
        d[0]; self.assertIsNot(d._con,con); self.assertIsNot(d._evidence_con,ev); self.assertIsNot(d._aug_generator,g)
        with patch('torch.utils.data.get_worker_info',return_value=SimpleNamespace(seed=456,id=0)):
            d[0]; worker_g = d._aug_generator; d[0]; self.assertIs(d._aug_generator,worker_g)
        d[0]; self.assertIsNot(d._aug_generator,worker_g)
        ctx = multiprocessing.get_context('spawn'); q = ctx.Queue()
        proc = ctx.Process(target=spawned_sample,args=(d,q)); proc.start()
        metadata,tracks = q.get(timeout=45); proc.join(45)
        self.assertEqual(proc.exitcode,0); self.assertEqual(metadata,{'rn':7,'has_evidence':True})
        q.close()


if __name__ == '__main__': unittest.main()
