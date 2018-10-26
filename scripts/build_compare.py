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

def read_result(inResultTsvFilename, dictThreshold, displayname = None):
    
    dictDisplayName = {}
    if displayname != None:
        with open(displayname, 'r') as f:
            lines = f.readlines()
            for line in lines:
                cols = [x.strip() for x in line.split('\t')]
                dictDisplayName[cols[0]] = cols[1]

    dictResult = {}
    with open(inResultTsvFilename, 'r') as f_in_tsv:
        lines = f_in_tsv.readlines()
        for line in lines:
            cols = [x.strip() for x in line.split('\t')]
            img_id = cols[0]

            labels = []
            raw_labels = json.loads(cols[1])

            if displayname != None:
                for item in raw_labels:
                    if item["class"] in dictDisplayName:
                        item["class"] = dictDisplayName[item["class"]]
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
        inLeftThresholdFilename=None,
        inRightThresholdFilename=None,
        inLeftLabel="left",
        inRightLabel="right",
        leftDisplay = None):

    dictLeftThreshold = {}
    if inLeftThresholdFilename != None:
        dictLeftThreshold = read_threshold(inLeftThresholdFilename)

    dictRightThreshold = {}
    if inRightThresholdFilename != None:
        dictRightThreshold = read_threshold(inRightThresholdFilename)

    dictLeftResult = read_result(inLeftResultTsvFilename, dictLeftThreshold, leftDisplay)
    dictRightResult = read_result(inRightResultTsvFilename, dictRightThreshold)

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
                    str(cnt)+"_"+inLeftLabel, json.dumps(dictLeftResult[img_id]), cols[-1]))
                if img_id in dictRightResult:
                    f_out_tsv.write("{}\t{}\t{}\n".format(
                        str(cnt)+"_"+inRightLabel, json.dumps(dictRightResult[img_id]), cols[-1]))
                else:
                    f_out_tsv.write("{}\t{}\t{}\n".format(
                        str(cnt)+"_Missed", [], cols[-1]))
            line = f_in_tsv.readline()
            cnt = cnt + 1
            if cnt % 100 == 0:
                print cnt

    f_out_tsv.close()
    print 'Total lines=' + str(cnt)
    print 'Finished building compare result.'


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Object Detection comparision')
    parser.add_argument('--left_tsv', required=True, type=str, help='left result tsv')
    parser.add_argument('--right_tsv', required=True, help='right result tsv')
    parser.add_argument('--image_tsv', required=True, help='image tsv')
    parser.add_argument('--target_tsv', required=False, default='test.tsv', help="target tsv")
    parser.add_argument('--left_threshold', required=False, default=None, help="left side threshold filename")
    parser.add_argument('--right_threshold', required=False, default=None, help="right side threshold filename")
    parser.add_argument('--left_tag', required=False, default='', help="left side tag")
    parser.add_argument('--right_tag', required=False, default='', help="right side tag")
    
    args = parser.parse_args()
    return args  

if __name__ == '__main__':
    args = parse_args()

    # build_side_by_side_compare('./data/Compare_V14.2_vs_Google_Instagram1K/instagram.v14.2.tsv',
    #     './data/Compare_V13.1_vs_Google_Instagram1K/instagram.google.tsv',
    #     './data/Top100Instagram-GUID/test.tsv',
    #     './data/Compare_V14.2_vs_Google_Instagram1K/test.tsv',
    #     None,
    #     None,
    #     'MSFT_v14.2',
    #     'Google')

    build_side_by_side_compare(args.left_tsv,
        args.right_tsv,
        args.image_tsv,
        args.target_tsv,
        args.left_threshold,
        args.right_threshold,
        args.left_tag,
        args.right_tag)