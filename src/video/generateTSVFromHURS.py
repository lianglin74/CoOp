# this script can be used to generate TSV files from the downloaded UHRS labeling results files.
from evaluation.analyze_task import analyze_draw_box_task

import json
import sys

# read the map file


def read_map_file(map_file_name):
    id_url_lists = []
    with open(map_file_name, 'r') as file:  # Use file to refer to the file object
        for line in file:
            strList = line.split()
            id_url_lists.append((strList[0], strList[1]))

    print(len(id_url_lists))
    return id_url_lists

# write to tsv using the url2ans_map
# def write_results_to_tsv(result_files, result_file_type, outfile_res):


def read_file_to_list(file_name):
    res_lists = []
    with open(file_name, 'r') as file:  # Use file to refer to the file object
        data = file.read()
        res_lists = data.split()

    return res_lists


def getResultsToTSV(map_file_name, result_list_file_name, outfile_res, tsvfile):
    id_url_lists = read_map_file(map_file_name)

    print(id_url_lists[0])

    result_file_type = 'uhrs'
    result_files = read_file_to_list(result_list_file_name)

    url2ans_map = analyze_draw_box_task(
        result_files, result_file_type, outfile_res)

    # write to tsv in original order
    with open(tsvfile, 'w') as file:
        for (id, url) in id_url_lists:
            # if url in url2ans_map:
            label = url2ans_map.get(url, "NotDone")
            file.write(id + "\t" + json.dumps(label)+"\n")


def generateCBA_0():
    map_file_name = "/raid/data/ChinaMobileVideoLabeling1/train.key.url.tsv"
    result_list_file_name = '/mnt/ivm_server2_iris/ChinaMobile/Video/uhrs/download/processed/filelist.txt'
    outfile_res = 'tmp_result.tsv'
    tsvfile = 'result.tsv'

    getResultsToTSV(map_file_name, result_list_file_name, outfile_res, tsvfile)


def generateCBA_1():
    map_file_name = "/raid/data/ChinaMobileVideoCBA_video_1/train.key.url.tsv"
    result_list_file_name = '/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/uhrs/download/filelist.txt'
    outfile_res = 'tmp_result_CBA_1.tsv'
    tsvfile = 'result_CBA_1.tsv'

    getResultsToTSV(map_file_name, result_list_file_name, outfile_res, tsvfile)


def generateCBA_2_part0_1():
    map_file_name = "/raid/data/ChinaMobileVideoCBA_video_2/train.key.url.tsv"
    result_list_file_name = '/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/uhrs/download/filelist.txt'
    outfile_res = 'tmp_result_CBA_2_p01.tsv'
    tsvfile = 'CBA_video_2_p01.label.tsv'

    getResultsToTSV(map_file_name, result_list_file_name, outfile_res, tsvfile)


def test_read_file_to_list():
    result_list_file_name = '/mnt/ivm_server2_iris/ChinaMobile/Video/uhrs/download/processed/filelist.txt'
    result_files = read_file_to_list(result_list_file_name)
    print("res_lists:", result_files)


if __name__ == '__main__':
    # generateCBA_2_part0_1()
    # test_read_file_to_list()
    if len(sys.argv) == 4:
        map_file_name = sys.argv[1]
        result_list_file_name = sys.argv[2]
        tsvfile = sys.argv[3]
        outfile_res = "url_as_id." + tsvfile
        getResultsToTSV(map_file_name, result_list_file_name,
                        outfile_res, tsvfile)
    else:
        print("Please check arguments")
