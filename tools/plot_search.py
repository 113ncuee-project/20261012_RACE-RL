"""Plot actual exported records, refusing missing/empty input or invented data."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--label",default="Toy diagnostic — not a formal benchmark")
    args = parser.parse_args()
    rows = json.loads(args.trace.read_text())
    if not rows or any("evaluations" not in r or "best_objective" not in r for r in rows):
        parser.error("nonempty actual exported search trace required")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(6,3.5),constrained_layout=True)
    ax.step([1]+[r["evaluations"] for r in rows],[1]+[r["best_objective"] for r in rows],where="post")
    ax.set(xlabel="Charged analytical mapping evaluations",ylabel="Best normalized EDP",title=args.label)
    ax.grid(alpha=.25)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out,dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
