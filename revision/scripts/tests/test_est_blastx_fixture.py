"""Hand-derived traceback contracts. These are NOT a substitute for the engine fixture."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'evidence/est_blastx_fixture.py'


class TracebackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if SCRIPT.exists():
            spec = importlib.util.spec_from_file_location('fixture', SCRIPT)
            cls.m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.m)

    def setUp(self):
        self.assertTrue(SCRIPT.exists(), 'real-engine fixture/replay not implemented')

    def row(self, dna, btop, cigar, qg, sg, translated, reverse=False):
        return ['q', str(len(dna)), 's', '3', str(len(dna) if reverse else 1),
                str(1 if reverse else len(dna)), '1', '3', '3', '1e-9', '50',
                '-1' if reverse else '1', btop, cigar, qg, sg, translated]

    def test_replays_both_frameshift_directions_and_reverse_complement(self):
        for dna, btop, cigar, qg, net in (
            ('ATGAAACGCT', '2\\-1', '2M1\\1M', 'MK\\A', 1),
            ('ATGAAGCT', '2/-1', '2M1/1M', 'MK/A', -1),
        ):
            row = self.row(dna, btop, cigar, qg, 'MK-A', 'MKA')
            self.assertEqual(self.m.replay(row, dna, 'MKA')['net_frameshift_nt'], net)
            rc = self.m.reverse_complement(dna)
            row[1], row[4], row[5], row[11] = str(len(rc)), str(len(rc)), '1', '-1'
            self.assertEqual(self.m.replay(row, rc, 'MKA')['net_frameshift_nt'], net)

    def test_replays_inframe_insertions_deletions(self):
        cases = [('ATGAAAGGGGCT', '2G-1', '2M1I1M', 'MKGA', 'MK-A', 'MKGA'),
                 ('ATGGCT', '1-K1', '1M1D1M', 'M-A', 'MKA', 'MA')]
        for dna, btop, cigar, qg, sg, translated in cases:
            self.assertEqual(self.m.replay(self.row(dna, btop, cigar, qg, sg, translated), dna, 'MKA')['net_frameshift_nt'], 0)

    def test_rejects_corrupted_btop_translation_endpoint_and_cigar(self):
        dna = 'ATGAAACGCT'
        row = self.row(dna, '2\\-1', '2M1\\1M', 'MK\\A', 'MK-A', 'MKA')
        for column, value in ((12, '2/-1'), (12, '2\\'), (5, '9'), (13, '3M'), (14, 'MK\\G'), (16, 'MKG')):
            bad = row.copy()
            bad[column] = value
            with self.assertRaises(ValueError):
                self.m.replay(bad, dna, 'MKA')

    def test_fixture_has_required_cases_and_species_sources(self):
        data = self.m.fixture_data()
        self.assertEqual(set(data['queries']), {'ordinary', 'reverse', 'plus1', 'minus1', 'insertion', 'deletion'})
        self.assertEqual(len(data['protein']), 180)
        self.assertEqual(len(data['queries']['plus1']), 541)
        self.assertEqual(len(data['queries']['minus1']), 539)
        self.assertEqual(data['queries']['reverse'], self.m.reverse_complement(data['queries']['ordinary']))


if __name__ == '__main__':
    unittest.main()
