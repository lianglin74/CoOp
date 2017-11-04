from __future__ import print_function
import os
import sys
import pandas as pd
import argparse
import textwrap
import json
import matplotlib.pyplot as plt

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)

from wordtree import WordTree


def tree_stats(t):
    """Find statistics of the tree
    :type t: WordTree
    :rtype: pd.DataFrame
    """
    stats = {
        'descendants': [],
        'children': [],
        'used': [],
        'height': []
    }
    for node in t:
        for k in stats.keys():
            val = getattr(node, k)
            if isinstance(val, tuple) or isinstance(val, list):
                val = len(val)
            stats[k].append(val)

    return pd.DataFrame(stats)


def annotate(t, tsv):
    """Annotate a tree with labels from TSV
    :type t: WordTree
    :type tsv: string
    """
    for path in tsv:
        labels_idx = None
        with open(path) as tsv_f:
            for line in tsv_f:
                elems = line.split(sep='\t')
                labels = None
                if labels_idx is not None:
                    labels = json.loads(elems[labels_idx])
                else:
                    for labels_idx, elem in enumerate(elems):
                        try:
                            labels = json.loads(elem)
                            break
                        except json.JSONDecodeError:
                            pass
                for label in labels:
                    if 'class' not in label:
                        continue
                    t[label['class']].used += 1


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Analyze a wordtree file (optionally with TSV files).',
        epilog=textwrap.dedent('''Example:
wt_stat.py d:/data/imagenet/9k.tree -p -t d:/data/imagenet/training.tsv
'''))

    parser.add_argument('-t', '--tsv', action='append', required=False, default=[],
                        help='TSV file used in generating the tree (can be specified multiple times)')
    parser.add_argument('-p', '--plot', action='store_true', required=False,
                        help='If should plot the stats')
    parser.add_argument('tree_path', metavar='PATH', help='path to the tree')

    if len(sys.argv) == 1:
        parser.print_help()
        raise Exception("Required input not provided")

    args = parser.parse_args()
    t = WordTree(args.tree_path)
    annotate(t, args.tsv)

    # Find the stats
    df = tree_stats(t)

    print('Non-leaf node stats:')
    print(df[df['children'] > 0].describe())
    if args.plot:
        axs = df.plot.line(subplots=True)
        fig = axs[0].get_figure()
        fig.canvas.set_window_title('Stats')

        # Nodes that can be pruned
        sdf = df[(df['children'] == 1) & (df['used'] == 0)].loc[:, (df.columns != 'used') &
                                                                   (df.columns != 'children')]

        if not sdf.empty:
            axs = sdf.plot.line(subplots=True)
            fig = axs[0].get_figure()
            fig.canvas.set_window_title('Zero used single child nodes')

        plt.show()

    return t, df

if __name__ == '__main__':
    tree, statsdf = main()
