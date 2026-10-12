"""Offline checks only. Uses the retained, controller-verified one-record fixture."""
import copy
import unittest
from pathlib import Path
import recover_metadata as r

class ParserChecks(unittest.TestCase):
    def setUp(self):
        self.members, _ = r.load_members()
        self.raw = (Path(__file__).resolve().parent.parent / 'JK596275.1.genbank.candidate').read_bytes()
        self.expected = {'JK596275.1': self.members['JK596275.1']}

    def test_exact_and_multiline(self):
        rows = r.validate_extract(self.raw, self.expected)
        self.assertEqual(rows[0]['sequence_length'], 648)
        self.assertIn('\n', rows[0]['qualifiers']['clone_lib'][0])
        self.assertIn('Library is normalized.', rows[0]['qualifiers']['note'][0])
        self.assertEqual(rows[0]['biosample'], ['SAMN00736318'])
        self.assertEqual(rows[0]['bioproject'], [])

    def test_fail_closed(self):
        cases = [self.raw + self.raw, self.raw.replace(b'JK596275.1', b'JK596275.2'),
                 self.raw.replace(b'attgagtaaa', b'cttgagtaaa'), self.raw[:-4],
                 b'ERROR\n' + self.raw, self.raw.replace(b'648 bp', b'649 bp')]
        for raw in cases:
            with self.subTest(raw=raw[:70]):
                with self.assertRaises(ValueError):
                    r.validate_extract(raw, self.expected)
        with self.assertRaises(ValueError):
            r.validate_extract(self.raw, self.members)
        bad = copy.deepcopy(self.expected)
        bad['JK596275.1']['original']['sequence_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            r.validate_extract(self.raw, bad)

if __name__ == '__main__':
    unittest.main()
