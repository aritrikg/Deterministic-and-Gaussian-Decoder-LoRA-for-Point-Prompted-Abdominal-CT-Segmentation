"""Build a self-contained runtime notebook below Kaggle's1 MB source cap."""
import base64, hashlib, io, json, lzma, sys, tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT.parent/'output'
OUTPUT.mkdir(parents=True,exist_ok=True)
files={}
for folder in ('base_v4','v5','evidence','recent_accounting'):
 for p in sorted((ROOT/folder).rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':files[str(p.relative_to(ROOT))]=p.read_bytes()
for name in ('README.md','REFERENCES.md','CLAIMS_AND_REMAINING.md','KAGGLE_SIZE_FIX.md','notebook_entry.py','runtime_launch.py','worker_entry.py'):
 files[name]=(ROOT/name).read_bytes()
manifest={name:hashlib.sha256(data).hexdigest() for name,data in sorted(files.items())}
files['RUNTIME_MANIFEST.json']=json.dumps(manifest,indent=2).encode()
stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w',format=tarfile.USTAR_FORMAT) as bundle:
 for name,data in sorted(files.items()):
  info=tarfile.TarInfo(name);info.size=len(data);info.mtime=0;info.mode=0o644
  bundle.addfile(info,io.BytesIO(data))
compressed=lzma.compress(stream.getvalue(),preset=9)
payload_sha=hashlib.sha256(compressed).hexdigest()
nb=json.loads((ROOT/'tools/notebook_template.ipynb').read_text())
bootstrap=(ROOT/'tools/embedded_runtime.py').read_text()+'\n\n'+(
 '_RUNTIME_XZ_B64 = '+repr(base64.b64encode(compressed).decode())+'\n'
 '_RUNTIME_ARCHIVE_SHA256 = '+repr(payload_sha)+'\n'
 'SOURCE_ROOT = install_embedded_runtime(_RUNTIME_XZ_B64, _RUNTIME_ARCHIVE_SHA256, Path("/kaggle/working"))\n'
 'sys.path[:0] = [str(SOURCE_ROOT), str(SOURCE_ROOT/"base_v4"), str(SOURCE_ROOT/"base_v4/source")]\n'
 'print("Source ready; no companion source dataset is required.")\n')
for cell in nb['cells']:
 if cell['cell_type']=='code':
  if '# SOURCE_LOADER_GENERATED_BY_BUILDER' in ''.join(cell['source']):cell['source']=bootstrap.splitlines(True)
  cell['execution_count']=None;cell['outputs']=[]
raw=json.dumps(nb,ensure_ascii=False,indent=1).encode()
if len(raw)>=900_000:raise ValueError(f'Notebook is {len(raw)} bytes; keep below900,000 for margin under1 MB')
path=OUTPUT/'BALoRA_Full_Research_Kaggle.ipynb';path.write_bytes(raw)
print(path,'| bytes:',len(raw),'| verified runtime files:',len(manifest))
