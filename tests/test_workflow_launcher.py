"""Exercise workspace isolation without invoking any scientific driver."""
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('workflow_launcher', Path(__file__).resolve().parents[1]/'pipeline/run.py')
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)

class WorkflowLauncher(unittest.TestCase):
    def test_plan_does_not_execute_or_write_and_missing_inputs_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            research = root/'research'; research.mkdir()
            (research/'driver.py').write_text('raise RuntimeError("must not run on plan")')
            recipe = dict(id='fixture', description='test', entrypoint='driver.py', inputs=['input.txt'], outputs=['output.txt'])
            with patch.object(launcher, 'RESEARCH', research):
                before = list(root.rglob('*'))
                result = launcher.plan(recipe)
                self.assertFalse(result['ready_to_prepare'])
                self.assertEqual(before, list(root.rglob('*')))
                with self.assertRaises(ValueError):
                    launcher.prepare(recipe, root/'run')
                self.assertFalse((root/'run').exists())

    def test_preparation_copies_inputs_and_detects_changed_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); research = root/'research'; research.mkdir()
            (research/'driver.py').write_text('from pathlib import Path\nPath("output.txt").write_text(Path("input.txt").read_text())\n')
            ext = root/'external'; ext.mkdir(); (ext/'input.txt').write_text('original')
            recipe = dict(id='fixture', description='test', entrypoint='driver.py', inputs=['input.txt'], outputs=['output.txt'])
            with patch.object(launcher, 'RESEARCH', research):
                launcher.prepare(recipe, root/'run', ext)
                self.assertFalse((root/'run/input.txt').is_symlink())
                self.assertFalse((root/'run/output.txt').exists())
                with self.assertRaises(ValueError):
                    launcher.prepare(recipe, root/'run', ext)
                (root/'run/input.txt').write_text('changed')
                with self.assertRaises(ValueError):
                    launcher.execute(root/'run')
                self.assertEqual((ext/'input.txt').read_text(), 'original')

    def test_explicit_run_receipt_and_no_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); research = root/'research'; research.mkdir()
            (research/'driver.py').write_text('from pathlib import Path\nPath("output.txt").write_text("done")\n')
            recipe = dict(id='fixture', description='test', entrypoint='driver.py', inputs=[], outputs=['output.txt'])
            with patch.object(launcher, 'RESEARCH', research):
                launcher.prepare(recipe, root/'run')
                result = launcher.execute(root/'run')
                self.assertEqual(result['state'], 'COMPLETED')
                self.assertTrue(result['inputs_unchanged'])
                with self.assertRaises(ValueError):
                    launcher.execute(root/'run')

    def test_manifest_path_relocation_preserves_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); research = root/'research'; research.mkdir()
            (research/'driver.py').write_text('pass')
            ext = root/'external'; (ext/'sources').mkdir(parents=True)
            (ext/'sources/a.npz').write_bytes(b'fixture')
            original = json.dumps([{'source_path':'/old/work/sources/a.npz','mass':1000000}])
            (ext/'manifest.json').write_text(original)
            recipe = dict(id='fixture', description='test', entrypoint='driver.py', inputs=['manifest.json','sources/a.npz'], outputs=[], input_json_relocations=[dict(file='manifest.json',array_key=None,field='source_path',target_directory='sources')])
            with patch.object(launcher, 'RESEARCH', research):
                result = launcher.prepare(recipe, root/'run', ext)
                row = json.loads((root/'run/manifest.json').read_text())[0]
                self.assertEqual(row, {'source_path':str((root/'run/sources/a.npz').resolve()),'mass':1000000})
                self.assertEqual((ext/'manifest.json').read_text(), original)
                self.assertNotEqual(result['inputs']['manifest.json']['sha256'],result['inputs']['manifest.json']['original_sha256'])

if __name__ == '__main__':
    unittest.main()
