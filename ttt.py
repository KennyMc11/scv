from pathlib import Path
import sys

site_packages = Path(sys.prefix) / "Lib" / "site-packages"
nvrtc_bin = site_packages / "nvidia" / "cuda_nvrtc" / "bin"

if nvrtc_bin.is_dir():
    for f in nvrtc_bin.glob("*.dll"):
        print(f"  {f.name}")
else:
    print("nvrtc_bin не найден:", nvrtc_bin)