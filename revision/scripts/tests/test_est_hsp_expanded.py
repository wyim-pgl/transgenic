"""CPU-only truth fixtures: these rows are NOT engine execution evidence."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'evidence/est_hsp_expanded.py'


def load():
    spec = importlib.util.spec_from_file_location('expanded', MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TruthTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.exists(), 'new independent interpreter is missing')
        self.m = load()

    def row(self, dna, q, s, btop, cigar, frame=1, offset=0, col9=None):
        protein = s.replace('-', '')
        end = len(dna) - offset
        qs, qe = offset + 1, len(dna)
        if frame < 0:
            qs, qe = len(dna) - offset, 1
        return ['q', str(len(dna)), 'p', str(len(protein)), str(qs), str(qe),
                '1', str(len(protein)), str(col9 or len(q.replace('\\', '').replace('/', ''))),
                '1e-20', '100', str(frame), btop, cigar, q, s,
                q.replace('-', '').replace('\\', '').replace('/', '')], protein

    def test_both_shift_signs_on_all_six_frames_and_wrong_sign(self):
        for offset in range(3):
            for sign in (1, -1):
                for core, q, btop, cigar in [('ATGCGCT', 'M\\A', '1\\-1', '1M1\\1M'),
                                             ('ATGCT', 'M/A', '1/-1', '1M1/1M')]:
                    oriented = 'C' * offset + core
                    dna = oriented if sign > 0 else self.m.rc(oriented)
                    row, protein = self.row(dna, q, 'M-A', btop, cigar,
                                            sign * (offset + 1), offset)
                    result = self.m.interpret(row, dna, protein)
                    self.assertEqual(result['counts'], dict(M=2, I=0, D=0, F=1))
                    self.assertEqual(result['blocks'], [1, 1])
                    with self.assertRaises(ValueError):
                        self.m.interpret(row, dna, protein, shift_sign=-1)

    def test_insertion_deletion_and_mismatch(self):
        row, p = self.row('ATGGCTTGG', 'MA-W', 'M-QF', '1A--QWF', '1M1I1D1M')
        r = self.m.interpret(row, 'ATGGCTTGG', p)
        self.assertEqual(r['counts'], dict(M=2, I=1, D=1, F=0))
        self.assertEqual(r['metrics']['query_residues'], 3)
        self.assertEqual(r['metrics']['paired'], 2)
        row[16] = 'MAF'
        with self.assertRaises(ValueError):
            self.m.interpret(row, 'ATGGCTTGG', p)

    def test_multiple_canceling_shifts(self):
        row, p = self.row('ATGCGCTGG', 'M\\A/W', 'M-A-W', '1\\-1/-1', '1M1\\1M1/1M')
        r = self.m.interpret(row, 'ATGCGCTGG', p)
        self.assertEqual(r['blocks'], [1, 1, 1])
        self.assertEqual([s['delta'] for s in r['shifts']], [1, -1])

    def test_boundary_metrics_and_unknown_reported_length(self):
        for n in (29, 30, 31):
            row, p = self.row('GCT' * n, 'A' * n, 'A' * n, str(n), f'{n}M')
            r = self.m.interpret(row, 'GCT' * n, p)
            self.assertEqual(r['thresholds']['reported'], n >= 30)
            row[8] = '999'
            r = self.m.interpret(row, 'GCT' * n, p)
            self.assertEqual(r['col9_matches'], [])

    def test_short_shift_threshold_and_serialization_failures(self):
        for n in (29, 30, 31):
            dna = 'ATG'*14 + 'C' + 'GCT'*(n-14)
            row, p = self.row(dna, 'M'*14 + '\\' + 'A'*(n-14),
                              'M'*14 + '-' + 'A'*(n-14),
                              f'14\\-{n-14}', f'14M1\\{n-14}M')
            result = self.m.interpret(row, dna, p)
            self.assertEqual(result['metrics']['columns_including_F'], n+1)
            self.assertEqual(result['thresholds']['reported'], n >= 30)
            for column, bad in [(5, '3'), (11, '2'), (12, '99'), (13, '99M'),
                                (9, 'nan'), (15, 'Z' + row[15][1:])]:
                broken = row.copy()
                broken[column] = bad
                with self.assertRaises(ValueError):
                    self.m.interpret(broken, dna, p)

    def test_shift_block_boundaries_both_strands(self):
        for n in (29, 30, 31):
            for marker, middle in [('\\', 'C'), ('/', '')]:
                core = 'ATG' * 29 + middle + 'GCT' * n
                if marker == '/':
                    core = 'ATG' * 29 + ('GCT' * n)[1:]
                for sign in (1, -1):
                    dna = core if sign > 0 else self.m.rc(core)
                    row, p = self.row(dna, 'M'*29 + marker + 'A'*n,
                                      'M'*29 + '-' + 'A'*n,
                                      f'29{marker}-{n}', f'29M1{marker}{n}M', sign)
                    r = self.m.interpret(row, dna, p)
                    self.assertEqual(r['blocks'], [29, n])
                    self.assertEqual(r['block_thresholds'], [False, n >= 30])
                    self.assertTrue(r['thresholds']['reported'])
                    self.assertFalse(r['each_block_ge30'])

    def test_manifest_bounded_and_ledger_reconstructs_dna(self):
        cases = self.m.make_cases()
        self.assertEqual(len(cases), 144)
        self.assertEqual({c['expected_frame'] for c in cases}, {-3, -2, -1, 1, 2, 3})
        for c in cases:
            self.m.validate_case(c)
            dna = c['baseline_dna']
            for edit in c['edits']:
                at = edit['at']
                self.assertEqual(dna[at:at + len(edit['deleted'])], edit['deleted'])
                dna = dna[:at] + edit['inserted'] + dna[at + len(edit['deleted']):]
            self.assertEqual(dna, c['oriented_dna'])
            self.assertGreaterEqual(len(dna), 100)
            self.assertEqual(c['dna'], dna if c['expected_frame'] > 0 else self.m.rc(dna))

    def test_ledger_tampering_rejected(self):
        case = self.m.make_cases()[18]  # first deletion family
        self.m.validate_case(case)
        case['edits'][0]['deleted'] = 'AAA'
        with self.assertRaises(ValueError):
            self.m.validate_case(case)

    def test_standard_code_known_residues(self):
        for codon, amino in [('ATG', 'M'), ('GCT', 'A'), ('TGG', 'W'),
                             ('TAA', '*'), ('TAG', '*'), ('TGA', '*'), ('AGC', 'S')]:
            self.assertEqual(self.m.CODE[codon], amino)
        for amino, codon in self.m.ENCODE.items():
            self.assertEqual(self.m.CODE[codon], amino)

    def test_boundary_indels_separate_paired_and_query(self):
        for n in (29, 30):
            dna = 'ATG'*14 + 'GCT' + 'ATG'*(n-14)
            row, p = self.row(dna, 'M'*14 + 'A' + 'M'*(n-14),
                              'M'*14 + '-' + 'M'*(n-14),
                              f'14A-{n-14}', f'14M1I{n-14}M')
            r = self.m.interpret(row, dna, p)
            self.assertEqual(r['metrics']['paired'], n)
            self.assertEqual(r['metrics']['query_residues'], n+1)
            dna = 'ATG'*n
            row, p = self.row(dna, 'M'*14 + '-' + 'M'*(n-14),
                              'M'*14 + 'A' + 'M'*(n-14),
                              f'14-A{n-14}', f'14M1D{n-14}M')
            r = self.m.interpret(row, dna, p)
            self.assertEqual(r['metrics']['query_residues'], n)
            self.assertEqual(r['metrics']['reported'], n+1)

    def test_raw_first_and_missing_cases_never_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            raw = out / 'raw.tsv'
            raw.write_text('broken\trow\n')
            cases = self.m.make_cases()[:1]
            report = self.m.diagnose(raw, cases, out)
            self.assertEqual((out / 'raw-first.tsv').read_bytes(), raw.read_bytes())
            self.assertEqual(report['status'], 'INCONCLUSIVE')
            self.assertFalse(report['certified'])
            self.assertEqual(report['cases'][0]['status'], 'NOT_ELICITED')
            self.assertEqual(report['rows'][0]['status'], 'INVALID')

    def test_complete_diagnostic_still_never_certifies(self):
        case = self.m.make_cases()[0]
        protein = case['protein']
        row = [case['id'], str(len(case['dna'])), case['subject'], '29',
               str(case['start']+1), str(case['end']), '1', '29', '29',
               '1e-20', '100', '1', '29', '29M', protein, protein, protein]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            raw = out / 'raw.tsv'
            raw.write_text('\t'.join(row) + '\n')
            report = self.m.diagnose(raw, [case], out)
            self.assertEqual(report['status'], 'DIAGNOSTIC_COMPLETE_REVIEW_REQUIRED')
            self.assertFalse(report['certified'])
            self.assertFalse(report['production_authorized'])
            self.assertEqual(report['acceptance_policy'], 'UNRESOLVED')
            with self.assertRaises(FileExistsError):
                self.m.diagnose(raw, [case], out)

    def test_clipped_trace_is_not_negative_validation(self):
        c = self.m.make_cases()[0]
        result = {'counts': c['expected_counts'], 'blocks': c['expected_blocks'],
                  'shifts': [], 'oriented_start': c['start'] + 3,
                  'oriented_end': c['end'], 'subject_start': 1,
                  'subject_end': len(c['protein']), 'initial_frame': c['expected_frame']}
        self.assertFalse(self.m.elicited(result, c))


if __name__ == '__main__':
    unittest.main()
