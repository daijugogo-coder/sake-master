import hashlib
import zipfile
import pathlib

files = [
    "bandit.json",
    "pip-audit.json",
    "sbom.json",
    "requirements.txt",
    "main.py",
]

extra_dirs = ["templates", "static"]

zip_path = pathlib.Path("SECURITY_EVIDENCE.zip")

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for f in files:
        p = pathlib.Path(f)
        if p.exists():
            z.write(p.as_posix())
    for d in extra_dirs:
        p = pathlib.Path(d)
        if p.exists():
            for fp in p.rglob("*"):
                if fp.is_file():
                    z.write(fp.as_posix())

sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()
sha_path = pathlib.Path("SECURITY_EVIDENCE.sha256")
sha_path.write_text(f"{sha256}  {zip_path.name}\n", encoding="utf-8")

print("OK")
print("ZIP :", zip_path)
print("SHA256:", sha256)
