"""Packaging checks only: no GPU work or modifications to scientific source."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

MODULE = Path(__file__).with_name('kaggle_source_loader.py')
spec = importlib.util.spec_from_file_location('compact_source_loader', MODULE)
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)


class LoaderChecks(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory) / 'dataset' / 'nested_source'
        root.mkdir(parents=True)
        content = {'v5/module.py': b'VALUE = 17\n', 'base_v4/config.json': b'{"complete": true}\n'}
        manifest = {name: hashlib.sha256(data).hexdigest() for name, data in content.items()}
        raw = json.dumps(manifest, sort_keys=True).encode()
        for name, data in content.items():
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        (root / 'SOURCE_MANIFEST.json').write_bytes(raw)
        return root, content, raw, hashlib.sha256(raw).hexdigest()

    def test_extracted_kaggle_input(self):
        with tempfile.TemporaryDirectory() as d:
            source, content, raw, pin = self.fixture(d)
            result = loader.install_kaggle_source(Path(d) / 'dataset', Path(d) / 'working', pin)
            for name, data in content.items():
                self.assertEqual((result / name).read_bytes(), data)

    def test_nested_zip_input(self):
        with tempfile.TemporaryDirectory() as d:
            source, content, raw, pin = self.fixture(d)
            folder = Path(d) / 'zip_input'
            folder.mkdir()
            archive = folder / 'BALoRA_Full_Research_Package.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('some_folder/SOURCE_MANIFEST.json', raw)
                for name, data in content.items():
                    z.writestr('some_folder/' + name, data)
                z.writestr('notebook.ipynb', '{}')
            result = loader.install_kaggle_source(folder, Path(d) / 'working', pin)
            self.assertTrue((result / 'v5/module.py').is_file())
            self.assertFalse((result / 'notebook.ipynb').exists())

    def test_changed_source_rejected_before_install(self):
        with tempfile.TemporaryDirectory() as d:
            source, content, raw, pin = self.fixture(d)
            (source / 'v5/module.py').write_text('WRONG = True\n')
            target = Path(d) / 'working'
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                loader.install_kaggle_source(source, target, pin)
            self.assertFalse(target.exists())

    def test_wrong_version_has_actionable_message(self):
        with tempfile.TemporaryDirectory() as d:
            source, content, raw, pin = self.fixture(d)
            with self.assertRaisesRegex(FileNotFoundError, 'Add Input / Upload'):
                loader.install_kaggle_source(source, Path(d) / 'working', '0' * 64)

    def test_path_traversal_rejected(self):
        raw = json.dumps({'../escape.py': '0' * 64}).encode()
        with self.assertRaisesRegex(ValueError, 'Unsafe source manifest path'):
            loader._checked_manifest(raw, hashlib.sha256(raw).hexdigest())

    def test_incomplete_resume_snapshot_is_not_the_source_package(self):
        with tempfile.TemporaryDirectory() as d:
            source, content, raw, pin = self.fixture(d)
            folder = Path(d) / 'zip_input'
            folder.mkdir()
            with zipfile.ZipFile(folder / '000_resume.zip', 'w') as z:
                z.writestr('source_snapshot/SOURCE_MANIFEST.json', raw)
                z.writestr('source_snapshot/v5/module.py', content['v5/module.py'])
            with zipfile.ZipFile(folder / 'source.zip', 'w') as z:
                z.writestr('full/SOURCE_MANIFEST.json', raw)
                for name, data in content.items():
                    z.writestr('full/' + name, data)
            target = Path(d) / 'working'
            result = loader.install_kaggle_source(folder, target, pin)
            self.assertEqual((result / 'base_v4/config.json').read_bytes(), content['base_v4/config.json'])

    def test_explicit_manifest_path_and_cache_repair(self):
        with tempfile.TemporaryDirectory() as d:
            source, content, raw, pin = self.fixture(d)
            target = Path(d) / 'working'
            loader.install_kaggle_source(Path(d), target, pin, str(source / 'SOURCE_MANIFEST.json'))
            (target / 'v5/module.py').write_bytes(b'local accidental edit')
            loader.install_kaggle_source(Path(d), target, pin, str(source))
            self.assertEqual((target / 'v5/module.py').read_bytes(), content['v5/module.py'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
