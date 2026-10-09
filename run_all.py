import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def run(*args):
    print(f"\n$ {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


if __name__ == "__main__":
    workers = sys.argv[1] if len(sys.argv) > 1 else "1"
    run("experiments/01_sanity_check.py")
    run("experiments/run.py", "--workers", workers)
    run("experiments/analyze.py")
    run("experiments/report.py")
    run("figures/make_architecture.py")
    run("figures/make_journal_figures.py")
