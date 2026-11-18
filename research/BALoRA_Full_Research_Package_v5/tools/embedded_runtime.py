"""Verified, self-contained runtime installation; no external source input."""
import base64
import hashlib
import io
import json
import lzma
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def install_embedded_runtime(encoded, expected_archive_sha256, working_root):
    compressed = base64.b64decode(encoded, validate=True)
    if hashlib.sha256(compressed).hexdigest() != expected_archive_sha256:
        raise ValueError('Embedded runtime checksum mismatch; import the original notebook again')
    raw = lzma.decompress(compressed, memlimit=256 * 1024**2)
    if len(raw) > 32 * 1024**2:
        raise ValueError('Unexpected embedded runtime size')
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:') as bundle:
        members = bundle.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or len(names) > 2000:
            raise ValueError('Invalid runtime archive members')
        for member in members:
            path = PurePosixPath(member.name)
            if (not member.isfile() or not member.name or path.is_absolute()
                    or '..' in path.parts or '\\' in member.name):
                raise ValueError('Unsafe embedded runtime path')
        manifest = json.loads(bundle.extractfile('RUNTIME_MANIFEST.json').read())
        if set(names) != set(manifest) | {'RUNTIME_MANIFEST.json'}:
            raise ValueError('Runtime manifest coverage differs from archive')
        contents = {}
        for name, checksum in manifest.items():
            data = bundle.extractfile(name).read()
            if hashlib.sha256(data).hexdigest() != checksum:
                raise ValueError('Runtime source mismatch: ' + name)
            contents[name] = data
    contents['RUNTIME_MANIFEST.json'] = json.dumps(manifest, indent=2).encode()
    destination = (Path(working_root) / ('balora_submission_runtime_v5_' + expected_archive_sha256[:12])).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in contents.items():
        path = (destination / name).resolve()
        if not path.is_relative_to(destination):
            raise ValueError('Runtime destination escapes working folder')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == hashlib.sha256(data).hexdigest():
            continue
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
    print('Verified self-contained runtime:', len(manifest), 'files', flush=True)
    return destination
