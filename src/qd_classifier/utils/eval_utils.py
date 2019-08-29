import json
import collections

from qd.tsv_io import tsv_reader, tsv_writer

def calculate_confusion_matrix(tag_pred_file, outfile):
    gt2preds = collections.defaultdict(list)
    for parts in tsv_reader(tag_pred_file):
        #gt_label = json.loads(parts[1])["class"]
        #pred = parts[2].split(';')[0].split(':')[0]
        gt_label = json.loads(parts[0])['class']
        pred = max(json.loads(parts[1]), key=lambda r: r['conf'])['class']
        gt2preds[gt_label].append(pred)

    pred_correct_rates = []
    for gt_label in gt2preds:
        pred_labels = gt2preds[gt_label]
        pred_counts = collections.Counter(pred_labels)
        sorted_pred_counts = sorted([[pred, count] for pred, count in pred_counts.items()], key=lambda t: t[1], reverse=True)
        correct_rate = 0.0
        for pred, count in sorted_pred_counts:
            if pred == gt_label:
                correct_rate = float(count) / len(pred_labels)
                break
        pred_correct_rates.append([correct_rate, gt_label] + sorted_pred_counts)

    pred_correct_rates = sorted(pred_correct_rates, key = lambda t: t[0])
    tsv_writer(pred_correct_rates, outfile)
