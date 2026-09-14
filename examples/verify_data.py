"""Check all bundled numerical input and artwork hashes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = json.loads((ROOT/'data/checksums.json').read_text())
    for rel, expected in manifest.items():
        path = (ROOT/rel).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError(f'Invalid input path: {rel}')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Missing or changed input: {rel}')
    print(f'Checked {len(manifest)} input files: all SHA256 hashes match.')

if __name__ == '__main__':
    main()
