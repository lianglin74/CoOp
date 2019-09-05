import json
import collections
import os.path as op

from qd.tsv_io import tsv_reader, tsv_writer

def calculate_confusion_matrix(tag_pred_file):
    outfile = op.splitext(tag_pred_file)[0] + '.confusion.report'
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
    return outfile

def compare_confusion_matrix(confusion_reports):
    label2accs = collections.defaultdict(list)
    for i, report in enumerate(confusion_reports):
        for parts in tsv_reader(report):
            label = parts[1]
            acc = float(parts[0])
            if i > 0:
                assert(label in label2accs)
            label2accs[label].append(acc)
    label_accs = [[k] + v for k, v in label2accs.items()]
    label_accs = sorted(label_accs, key=lambda t: t[1])
    return label_accs

