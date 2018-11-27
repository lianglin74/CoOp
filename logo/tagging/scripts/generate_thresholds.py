from sklearn import metrics
import os
import argparse
import numpy as np

def construct_arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # necessary inputs
    parser.add_argument('--labelmap', default='', type=str, metavar='PATH',
                        help='path to labelmap', required = True)
    parser.add_argument('--results', default="", type=str, metavar='PATH',
                        help='path to prediction file', required = True)
    parser.add_argument('--gt', default='', type=str, metavar='PATH',
                        help='ground truth labels', required = True)
    parser.add_argument('--precisions', type=float, nargs='+', required = True)
    parser.add_argument('--output', default='pr_ap_result', metavar='PATH',
                        help='output folder path')

    return parser

def read_labelmap(labelmap_file):
    print("Reading labelmap....")
    labelmap = []
    with open(labelmap_file, "r") as labelmap_in:
        for line in labelmap_in:
            labelmap.append(line.strip())
    return labelmap

def read_predictions(prediction_file):
    print("Reading predictions....")
    predictions_by_label= dict()
    n = 0
    with open(prediction_file, "r") as prediction_in:
        for line in prediction_in:
            n = n + 1
            if n % 20000 == 0:
                print(str(n)+"...")
            parts = line.strip().split("\t")
            img_id = parts[0]
            pred_col = len(parts)-1
            predictions = parts[pred_col].split(";")
            for pred in predictions:
                temp = pred.split(":")
                label = temp[0]
                conf = temp[1]
                if label not in predictions_by_label:
                    predictions_by_label[label] = []
                predictions_by_label[label].append((img_id, conf))
    return predictions_by_label

def read_img_labels(label_file):
    print("Reading image labels....")
    imgid_by_label= dict()
    n = 0
    with open(label_file, "r") as label_in:
        for line in label_in:
            n = n + 1
            if n % 20000 == 0:
                print(str(n)+"...")
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            img_id = parts[0]
            labels = parts[1].split(',')
            for label in labels:
                if label not in imgid_by_label:
                    imgid_by_label[label] = set()
                imgid_by_label[label].add(img_id)

    return imgid_by_label

def calculate_PR_AP(labelmap, predictions_by_label, imgid_by_label):
    print("Calculating pr....")
    p_r_thres_by_label = dict()
    average_precision_by_label = dict()
    labels_not_in_val_set = []
    for label in labelmap:
        print("Processing label " + label +" ...")
        if label not in imgid_by_label:
            print("Label not in the validation set...")
            average_precision_by_label[label] = 0
            labels_not_in_val_set.append(label)
            continue

        if label not in predictions_by_label:
            print("No predictions made for this label...")
            average_precision_by_label[label] = 0
            continue

        img_ids_of_cur_label = imgid_by_label[label]
        predictions = predictions_by_label[label]
        prediction_is_right = np.array([1 if pred[0] in img_ids_of_cur_label else 0 for pred in predictions])
        confs = [np.exp(float(pred[1])) for pred in predictions]
        confs = np.array([x/(1+x) for x in confs])
        n_right_preds = np.sum(prediction_is_right)
        if n_right_preds > 0:
            precision, recall, thresholds = metrics.precision_recall_curve(prediction_is_right, confs)
            p_r_thres_by_label[label] = (precision, recall, thresholds)
        else:
            print("No right predictions for " + label)
            average_precision_by_label[label] = 0
            continue

        average_precision_by_label[label] = metrics.average_precision_score(prediction_is_right, confs)
        print("ap is " + str(average_precision_by_label[label]))

    return p_r_thres_by_label, average_precision_by_label, labels_not_in_val_set

def main():
    parser = construct_arg_parser()
    args = parser.parse_args()
    labelmap = read_labelmap(args.labelmap)
    predictions_by_label = read_predictions(args.results)
    imgid_by_label = read_img_labels(args.gt)

    pr, ap, labels_not_in_val_set = calculate_PR_AP(labelmap, predictions_by_label, imgid_by_label)

    if os.path.isdir(args.output) == False:
        os.makedirs(args.output)

    missing_label_file = os.path.join(args.output, "labels_not_in_val.tsv")
    with open(missing_label_file, "w") as missing_labels:
        for l in labels_not_in_val_set:
            missing_labels.write(l+'\n')


    for precision_point in args.precisions:
        precision_point = float(precision_point)
        print("Write thresholds + recall to disk for precision " + str(precision_point))
        thres_recall_by_label_file_path = os.path.join(args.output, str(precision_point) + ".tsv")
        with open(thres_recall_by_label_file_path, "w") as out:
            for label in labelmap:
                print("Write result for label " + label + " ...")
                if label not in pr:
                    out.write(label + "\t0.0\t1.0\n")
                    continue
                precs = pr[label][0]
                recalls = pr[label][1]
                threshs = pr[label][2]
                indices = np.where((precs > precision_point) & (recalls > 0.0))[0]

                if len(indices) == 0 or indices[0] == len(threshs):
                    out.write(label + "\t0.0\t1.0\n")
                    continue
                index = indices[0]
                out.write(label + "\t" + str(recalls[index]) + "\t" + str(threshs[index]) + "\n")

    ap_file_path = os.path.join(args.output, "ap.tsv")
    with open(ap_file_path, "w") as ap_out:
        for label in labelmap:
            if label not in ap:
                ap_out.write(label+"\t0.0\n")
            else:
                ap_out.write(label+"\t"+str(ap[label])+"\n")


if __name__ == '__main__':
    main()
