from pathlib import Path
import shutil

root = Path(__file__).parent / "records"
source = root / "e9"
for name in ("report.json", "source.json"):
    target = root / name
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(Path("e9") / name)
target = root / "align"
if target.is_symlink() or target.exists():
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target)
    else:
        target.unlink()
target.symlink_to(Path("e9") / "align", target_is_directory=True)
print("Prepared root paths used by the follow-up scripts.")
