#!/usr/bin/python
# coding:utf8
import os
import base64
import cv2
import sys
import json
import glob
import shutil
import argparse
import os.path as op

def read_result(inResultTsvFilename, dictThreshold, MinConf=0.5):
    dictResult = {}
    with open(inResultTsvFilename, 'r') as f_in_tsv:
        lines = f_in_tsv.readlines()
        for line in lines:
            cols = [x.strip() for x in line.split('\t')]
            img_id = cols[0]

            labels = []
            raw_labels = json.loads(cols[1])

            if inResultTsvFilename.find("google") != -1:
                labels = raw_labels
            else:
                if dictThreshold != None:
                    for item in raw_labels:
                        if item["class"] in dictThreshold:
                            if item["conf"] >= dictThreshold[item["class"]]:
                                if item["conf"] >= MinConf:
                                    labels.append(item)
                else:
                    labels = raw_labels

            dictResult[img_id] = labels
    return dictResult


def read_threshold(inThresholdFilename):
    dictThreshold = {}
    if inThresholdFilename != None:
        with open(inThresholdFilename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                cols = [x.strip() for x in line.split('\t')]
                term = cols[0]
                dictThreshold[term] = float(cols[1])
    return dictThreshold


def build_side_by_side_compare(
        inLeftResultTsvFilename,
        inRightResultTsvFilename,
        inImageTsvFilename,
        outTsvFilename,
        inLeftThresholdFilename,
        inRightThresholdFilename,
        inLeftLabel,
        inRightLabel,
        inLeftMinConf,
        inRightMinConf):

    dictLeftThreshold = {}
    if inLeftThresholdFilename != None:
        dictLeftThreshold = read_threshold(inLeftThresholdFilename)

    dictRightThreshold = {}
    if inRightThresholdFilename != None:
        dictRightThreshold = read_threshold(inRightThresholdFilename)

    dictLeftResult = read_result(inLeftResultTsvFilename, dictLeftThreshold, inLeftMinConf)
    dictRightResult = read_result(inRightResultTsvFilename, dictRightThreshold,inRightMinConf)

    print 'Start to write compare result . . .'
    f_out_tsv = open(outTsvFilename, 'w')

    with open(inImageTsvFilename, 'r') as f_in_tsv:
        line = f_in_tsv.readline()
        cnt = 0
        while line:
            cols = [x.strip() for x in line.split('\t')]
            img_id = cols[0]
            if img_id in dictLeftResult:
                f_out_tsv.write("{}\t{}\t{}\n".format(
                    img_id+"_"+inLeftLabel, json.dumps(dictLeftResult[img_id]), cols[-1]))
                if img_id in dictRightResult:
                    f_out_tsv.write("{}\t{}\t{}\n".format(
                        img_id+"_"+inRightLabel, json.dumps(dictRightResult[img_id]), cols[-1]))
                else:
                    f_out_tsv.write("{}\t{}\t{}\n".format(
                        img_id+"_Missed", [], cols[-1]))
            line = f_in_tsv.readline()

            cnt = cnt + 1
            if cnt%100 == 0:
                sys.stdout.write('#')
                sys.stdout.flush()

    print ('{}'.format(cnt))

    f_out_tsv.close()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Object Detection comparision')
    parser.add_argument('--left_tsv', required=True, type=str, help='left result tsv file')
    parser.add_argument('--right_tsv', required=True, help='right result tsv file')
    parser.add_argument('--image_tsv', required=True, help='image tsv file')
    parser.add_argument('--folder_out', required=True,  help="output folder name")
    parser.add_argument('--target_tsv', required=False, default='test.tsv', help="target tsv file name")
    parser.add_argument('--left_threshold', required=False, default=None, help="left side threshold filename")
    parser.add_argument('--right_threshold', required=False, default=None, help="right side threshold filename")
    parser.add_argument('--left_min_conf', required=False,  default=0.5, help="left side min confidence")
    parser.add_argument('--right_min_conf', required=False,  default=0.5, help="right side min confidence")
    parser.add_argument('--left_tag', required=False, default='', help="left side tag")
    parser.add_argument('--right_tag', required=False, default='', help="right side tag")
    
    args = parser.parse_args()
    return args  

if __name__ == '__main__':
    args = parse_args()

    build_side_by_side_compare(args.left_tsv,
        args.right_tsv,
        args.image_tsv,
        op.join(args.folder_out, args.target_tsv),
        args.left_threshold,
        args.right_threshold,
        args.left_tag,
        args.right_tag,
        args.left_min_conf,
        args.right_min_conf)