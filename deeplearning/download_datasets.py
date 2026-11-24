"""
Henter de to datasæt til skateboard trick-detektoren:

  1. sensui0731 / skateboarding-trick-dataset-120fps (Kaggle, CC BY 4.0)
     Træningsdata. Ollie, Shuvit, FrontShuvit, Kickflip, Heelflip (+ 3 klasser vi ikke bruger).
  2. LightningDrop / SkateboardML (GitHub)
     Ekstra eksternt testsæt. Kun Ollie og Kickflip.

Brug:
    pip install kagglehub
    python download_datasets.py                 # Dataset_Kaggle/ og SkateboardML-1.0/ i denne mappe
    python download_datasets.py --only kaggle   # eller --only github

Kaggle-login:
    kagglehub prøver først uden login. Hvis det fejler, så hent et API-token på
    kaggle.com -> Settings -> API -> "Create New Token" og læg kaggle.json i
    ~/.kaggle/ (Linux/Mac) eller C:\\Users\\<dig>\\.kaggle\\ (Windows).
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

KAGGLE_HANDLE = "sensui0731/skateboarding-trick-dataset-120fps"
GITHUB_REPO = "LightningDrop/SkateboardML"
VIDEO_EXT = {".mov", ".mp4", ".avi", ".mkv", ".webm", ".m4v"}


def count_videos(root: Path) -> None:
    """Printer antal videoklip per mappe, så man kan tjekke klassebalancen."""
    counts = {}
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in VIDEO_EXT:
            key = f.parent.relative_to(root).as_posix() or "."
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        print(f"  Ingen videofiler fundet under {root}")
        return
    total = sum(counts.values())
    for k in sorted(counts):
        print(f"  {k:<50} {counts[k]:>5}")
    print(f"  {'I alt':<50} {total:>5}")


def download_kaggle(target: Path) -> None:
    print(f"\n[Kaggle] {KAGGLE_HANDLE}")
    if target.exists() and any(target.iterdir()):
        print(f"  Findes allerede i {target}, springer over (slet mappen for at hente igen)")
        count_videos(target)
        return

    try:
        import kagglehub
    except ImportError:
        sys.exit("  kagglehub mangler. Kør: pip install kagglehub")

    try:
        cache_path = Path(kagglehub.dataset_download(KAGGLE_HANDLE))
    except Exception as e:
        sys.exit(
            f"  Download fejlede: {e}\n"
            "  Læg dit kaggle.json API-token i ~/.kaggle/ og prøv igen."
        )

    print(f"  Hentet til cache: {cache_path}")
    print(f"  Kopierer til {target} ...")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cache_path, target, dirs_exist_ok=True)
    count_videos(target)


def download_github(target: Path) -> None:
    print(f"\n[GitHub] {GITHUB_REPO}")
    if target.exists() and any(target.iterdir()):
        print(f"  Findes allerede i {target}, springer over (slet mappen for at hente igen)")
        count_videos(target)
        return

    target.parent.mkdir(parents=True, exist_ok=True)

    # Først git clone, det er hurtigst og giver mulighed for git pull senere
    if shutil.which("git"):
        print("  Kloner med git ...")
        r = subprocess.run(
            ["git", "clone", "--depth", "1", f"https://github.com/{GITHUB_REPO}.git", str(target)]
        )
        if r.returncode == 0:
            count_videos(target)
            return
        print("  git clone fejlede, prøver zip-download i stedet")
        shutil.rmtree(target, ignore_errors=True)

    # Fallback: zip af standard-branchen
    for branch in ("master", "main"):
        url = f"https://github.com/{GITHUB_REPO}/archive/refs/heads/{branch}.zip"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                zip_path = Path(tmp) / "repo.zip"
                print(f"  Henter {url} ...")
                urllib.request.urlretrieve(url, zip_path)
                with zipfile.ZipFile(zip_path) as z:
                    z.extractall(tmp)
                # Zip'en indeholder én topmappe, fx SkateboardML-master
                inner = next(p for p in Path(tmp).iterdir() if p.is_dir())
                shutil.move(str(inner), str(target))
            count_videos(target)
            return
        except Exception as e:
            print(f"  {branch}: {e}")

    sys.exit("  Kunne ikke hente SkateboardML. Tjek netforbindelsen eller hent zip'en manuelt fra GitHub.")


def main() -> None:
    p = argparse.ArgumentParser(description="Hent datasæt til skateboard trick-detektoren")
    p.add_argument("--root", type=Path, default=Path("."), help="Mappe datasættene lægges i (standard: denne mappe)")
    p.add_argument("--only", choices=["kaggle", "github"], help="Hent kun det ene datasæt")
    args = p.parse_args()

    root = args.root.resolve()
    print(f"Datamappe: {root}")

    if args.only in (None, "kaggle"):
        download_kaggle(root / "Dataset_Kaggle")
    if args.only in (None, "github"):
        download_github(root / "SkateboardML-1.0")

    print("\nFærdig.")


if __name__ == "__main__":
    main()
