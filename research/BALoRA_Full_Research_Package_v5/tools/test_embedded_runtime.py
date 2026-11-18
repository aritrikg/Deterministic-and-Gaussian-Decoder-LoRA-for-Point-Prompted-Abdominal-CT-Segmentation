"""Self-contained source packaging tests; no model training or data input."""
import base64,hashlib,importlib.util,io,json,lzma,tarfile,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('embedded_runtime',Path(__file__).with_name('embedded_runtime.py'))
loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)

def fixture(content=None,manifest=None,symlink=False):
 content=content if content is not None else {'v5/test_module.py':b'VALUE=17\n','base_v4/test.json':b'{}'}
 manifest=manifest if manifest is not None else {n:hashlib.sha256(b).hexdigest() for n,b in content.items()}
 stream=io.BytesIO()
 with tarfile.open(fileobj=stream,mode='w',format=tarfile.USTAR_FORMAT) as t:
  for name,data in {**content,'RUNTIME_MANIFEST.json':json.dumps(manifest).encode()}.items():
   item=tarfile.TarInfo(name);item.size=len(data);t.addfile(item,io.BytesIO(data))
  if symlink:
   item=tarfile.TarInfo('bad_link');item.type=tarfile.SYMTYPE;item.linkname='/tmp/elsewhere';t.addfile(item)
 raw=lzma.compress(stream.getvalue(),preset=6)
 return base64.b64encode(raw).decode(),hashlib.sha256(raw).hexdigest(),content

class EmbeddedChecks(unittest.TestCase):
 def test_no_source_dataset_needed(self):
  encoded,pin,content=fixture()
  with tempfile.TemporaryDirectory() as d:
   root=loader.install_embedded_runtime(encoded,pin,d)
   for name,data in content.items():self.assertEqual((root/name).read_bytes(),data)
 def test_archive_corruption_rejected(self):
  encoded,pin,_=fixture()
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex(ValueError,'archive|checksum'):loader.install_embedded_runtime(encoded,'0'*64,d)
 def test_file_mismatch_before_install(self):
  encoded,pin,_=fixture(manifest={'v5/test_module.py':'0'*64,'base_v4/test.json':hashlib.sha256(b'{}').hexdigest()})
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex(ValueError,'source mismatch'):loader.install_embedded_runtime(encoded,pin,d)
   self.assertEqual(list(Path(d).iterdir()),[])
 def test_unsafe_path_rejected(self):
  encoded,pin,_=fixture(content={'../outside.py':b'bad'})
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex(ValueError,'Unsafe'):loader.install_embedded_runtime(encoded,pin,d)
 def test_symlink_rejected(self):
  encoded,pin,_=fixture(symlink=True)
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex(ValueError,'Unsafe'):loader.install_embedded_runtime(encoded,pin,d)
 def test_manifest_coverage_rejected(self):
  encoded,pin,_=fixture(manifest={'v5/test_module.py':hashlib.sha256(b'VALUE=17\n').hexdigest()})
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex(ValueError,'coverage'):loader.install_embedded_runtime(encoded,pin,d)
 def test_reinstall_preserves_result_checkpoint(self):
  encoded,pin,content=fixture()
  with tempfile.TemporaryDirectory() as d:
   checkpoint=Path(d)/'results/last.pt';checkpoint.parent.mkdir();checkpoint.write_bytes(b'saved training state')
   root=loader.install_embedded_runtime(encoded,pin,d);(root/'v5/test_module.py').write_text('accidental edit')
   loader.install_embedded_runtime(encoded,pin,d)
   self.assertEqual((root/'v5/test_module.py').read_bytes(),content['v5/test_module.py'])
   self.assertEqual(checkpoint.read_bytes(),b'saved training state')
if __name__=='__main__':unittest.main(verbosity=2)
