import argparse

from qos_offloading.runner import run_jobs
from qos_offloading.studies import STUDIES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("studies", nargs="*", default=list(STUDIES))
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    jobs = []
    for name in args.studies:
        js = list(STUDIES[name]().values())
        print(f"{name}: {len(js)} jobs")
        jobs += js
    run_jobs(jobs, workers=args.workers)


if __name__ == "__main__":
    main()
