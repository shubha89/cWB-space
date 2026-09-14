"""Verify the archived driver bytes without importing research modules."""
from pathlib import Path
import hashlib
import json
import unittest

ROOT = Path(__file__).resolve().parents[1]

class ArchivedSources(unittest.TestCase):
    def test_original_source_hashes(self):
        records = json.loads((ROOT/'pipeline/SOURCE_MANIFEST.json').read_text())['files']
        self.assertGreater(len(records), 0)
        for record in records:
            path = ROOT/'pipeline/research'/record['path']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record['sha256'], record['path'])
            restored = path.read_bytes().replace(b'gpt6_astra_validation_scripts', b'.paper_agent')
            self.assertEqual(hashlib.sha256(restored).hexdigest(), record['original_sha256'], record['path'])

if __name__ == '__main__':
    unittest.main()
