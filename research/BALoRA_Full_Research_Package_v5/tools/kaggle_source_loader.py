"""Load a hash-pinned source tree from an attached ZIP or extracted Kaggle input.

Only files named by the verified source manifest are installed. Research
configuration, checkpoints, ledgers and outputs are never modified here.
"""
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def _sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _checked_manifest(raw, expected):
    if _sha_bytes(raw) != expected:
        return None
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or not manifest or len(manifest) > 3000:
        raise ValueError('Invalid pinned source manifest')
    for name, checksum in manifest.items():
        path = PurePosixPath(name)
        if not isinstance(name, str) or not name or '\\' in name or path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe source manifest path')
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise ValueError('Invalid source checksum')
    return manifest


def _install(raw, manifest, reader, destination):
    destination = Path(destination).resolve()
    # Verify every byte before writing executable files.
    content = {}
    total = 0
    for name, checksum in manifest.items():
        data = reader(name)
        total += len(data)
        if len(data) > 20 * 1024**2 or total > 50 * 1024**2:
            raise ValueError('Source package exceeds the expected source size')
        if _sha_bytes(data) != checksum:
            raise ValueError('Source checksum mismatch: ' + name)
        content[name] = data
    content['SOURCE_MANIFEST.json'] = raw
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in content.items():
        path = (destination / name).resolve()
        if not path.is_relative_to(destination):
            raise ValueError('Source destination escapes its directory')
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return destination


def install_kaggle_source(input_root, destination, expected_manifest_sha256, source_package_path=''):
    """Accept automatic Kaggle extraction and ZIP-as-file dataset layouts."""
    root = Path(source_package_path or input_root).expanduser()
    if not root.exists():
        raise FileNotFoundError('Source input path does not exist: ' + str(root))
    manifests = ([root] if root.is_file() and root.name == 'SOURCE_MANIFEST.json'
                 else sorted(root.rglob('SOURCE_MANIFEST.json')) if root.is_dir() else [])
    for manifest_path in manifests:
        raw = manifest_path.read_bytes()
        manifest = _checked_manifest(raw, expected_manifest_sha256)
        if manifest is None:
            continue
        source_root = manifest_path.parent.resolve()
        # Resumable result archives contain a partial source snapshot with the
        # same manifest but omit historical assets. Keep looking for the full
        # companion source input rather than treating that snapshot as complete.
        if not all((source_root / name).is_file() for name in manifest):
            continue
        def read_source(name):
            path = (source_root / name).resolve()
            if not path.is_relative_to(source_root):
                raise ValueError('Source input escapes the package')
            return path.read_bytes()
        result = _install(raw, manifest, read_source, destination)
        print('Verified source loaded from extracted input:', source_root, flush=True)
        return result
    archives = ([root] if root.is_file() and root.suffix.lower() == '.zip'
                else sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() == '.zip') if root.is_dir() else [])
    for archive in archives:
        try:
            with zipfile.ZipFile(archive) as bundle:
                names = bundle.namelist()
                for name in names:
                    if PurePosixPath(name).name != 'SOURCE_MANIFEST.json':
                        continue
                    info = bundle.getinfo(name)
                    if info.file_size > 1024**2:
                        continue
                    raw = bundle.read(name)
                    manifest = _checked_manifest(raw, expected_manifest_sha256)
                    if manifest is None:
                        continue
                    if len(names) != len(set(names)):
                        raise ValueError('Duplicate source ZIP member names')
                    prefix = str(PurePosixPath(name).parent)
                    prefix = '' if prefix == '.' else prefix + '/'
                    available = set(names)
                    if not all(prefix + rel in available for rel in manifest):
                        continue
                    def read_member(rel):
                        info = bundle.getinfo(prefix + rel)
                        if info.file_size > 20 * 1024**2:
                            raise ValueError('Unexpectedly large source member')
                        return bundle.read(info)
                    result = _install(raw, manifest, read_member, destination)
                    print('Verified source loaded from ZIP input:', archive, flush=True)
                    return result
        except zipfile.BadZipFile:
            continue
    raise FileNotFoundError(
        'Source package missing or from a different version. Download the updated '
        'BALoRA_Full_Research_Package.zip and attach it using Kaggle Add Input / Upload '
        '(suggested dataset name: balora-full-research-source-v5). Keep BTCV and the '
        'completed result inputs attached. ZIP files and automatically extracted '
        'datasets are both supported. Set SOURCE_PACKAGE_PATH to its exact path if needed.'
    )
