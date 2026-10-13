"""Plot completed seed outcomes and descriptive paired intervals, without significance claims."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();root=args.input
    rows=json.loads((root/'trial_results.json').read_text());analysis=json.loads((root/'analysis.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text());names=json.loads((root/'summary.json').read_text())['variants'];seeds=protocol['seeds']
    if len(rows)!=len(names)*len(seeds):raise RuntimeError('incomplete paired dataset')
    fig,axes=plt.subplots(1,2,figsize=(15,6),gridspec_kw={'width_ratios':[1,1.3]})
    colors=['#276FBF','#D45A26','#30866B']
    for i,seed in enumerate(seeds):
        values=[next(r['frozen_best_normalized_ppa_merit'] for r in rows if r['seed']==seed and r['variant']==name) for name in names]
        axes[0].scatter(values,np.arange(len(names))+(i-(len(seeds)-1)/2)*0.16,s=38,color=colors[i%3],label=f'seed {seed}',zorder=3)
    axes[0].set_yticks(range(len(names)),names);axes[0].invert_yaxis();axes[0].set_xlabel('Best normalized PPA merit (lower is better)')
    axes[0].set_title('Frozen-policy outcomes');axes[0].grid(axis='x',alpha=.2)
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(.27,.945),ncol=3,frameon=False)
    comparisons=analysis['comparisons'];labels=[]
    for i,c in enumerate(comparisons):
        mean=c['paired_mean_difference'];lo,hi=c['paired_bootstrap_ci95']
        axes[1].plot([lo,hi],[i,i],color='#276FBF',linewidth=2)
        axes[1].scatter([mean],[i],color='#D45A26',s=32,zorder=3)
        labels.append(f"{c['hypothesis']}: {c['left']} vs {c['right']}")
    axes[1].set_yticks(range(len(labels)),labels);axes[1].invert_yaxis();axes[1].axvline(0,color='gray',linestyle='--',linewidth=1)
    axes[1].set_title('Paired mean difference and descriptive bootstrap CI');axes[1].set_xlabel('Left minus right PPA merit; negative favors left');axes[1].grid(axis='x',alpha=.2)
    fig.suptitle(f'Toy development pilot: {len(seeds)} policy seeds, mesh compute counts 2–16',fontsize=14)
    fig.text(.5,.015,'Exploratory results only. Short training, one workload; no formal significance, CNN superiority or generalization claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.89));args.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out,dpi=180);plt.close(fig)


if __name__=='__main__':main()
