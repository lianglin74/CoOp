import json
from qd import tsv_io
from qd.process_tsv import onebb_in_list_of_bb
from qd.tsv_io import tsv_reader, tsv_writer

from tqdm import tqdm


def generate_vendor_labels_to_verify(predict_file, vendor_file, output_file, th=0.5,
                                     pred_to_gt=True):

    gt_key_rects = [(row[0], json.loads(row[1]))
                    for row in tqdm(tsv_reader(vendor_file))]

    #logging.info('loading pred')

    pred_key_to_rects = {}
    for row in tqdm(tsv_reader(predict_file)):
        key = row[0]
        rects = json.loads(row[1])
        maxBasketBallConf = th
        maxBasketRectIndex = -1
        filteredRects = []
        i = 0
        for r in rects:
            if r['class'] == 'basketball':
                if r['conf'] >= maxBasketBallConf:
                    maxBasketBallConf = r['conf']
                    maxBasketRectIndex = i
            elif r['conf'] > th:
                filteredRects.append(r)
            i += 1
        if maxBasketRectIndex != -1:
            filteredRects.append(rects[maxBasketRectIndex])

        #pred_key_to_rects[key] = [r for r in rects if r['conf'] > th]
        pred_key_to_rects[key] = filteredRects

    debug = False

    def gen_rows(pred_to_gt=True):
        #logging.info('start to writting')
        i = 0
        for key, gt_rects in tqdm(gt_key_rects):
            pred_rects = pred_key_to_rects.get(key)
            i += 1

            if debug:
                print("---", i,  key, gt_rects)

            if debug and i > 11:
                return
            if pred_rects is None and gt_rects is None:
                if debug:
                    print(str(i) + ": both empty")
                continue

            need_confirm = []
            if len(pred_rects) != len(gt_rects):
                if debug:
                    print(str(i) + ": size does not match")
                    print("prediction: ",  pred_rects)
                    print("vendor: ",  gt_rects)
                need_confirm.extend(gt_rects)
            else:
                if pred_to_gt:
                    findSuspect = False
                    for pr in pred_rects:
                        if not onebb_in_list_of_bb(pr, gt_rects):
                            findSuspect = True
                            break
                    if findSuspect:
                        need_confirm.extend(gt_rects)
                    else:
                        continue
                else:
                    for g in gt_rects:
                        if len(g) == 2:
                            assert 'class' in g and 'rect' in g
                            continue
                        if not onebb_in_list_of_bb(g, pred_rects):
                            need_confirm.append(g)
            yield i, key, json.dumps(need_confirm)

    tsv_writer(gen_rows(pred_to_gt=pred_to_gt), output_file)


def main():
    predict_file = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/CBA_video_2_p23.withLabel/model_iter_0030163.pt.CBAVideo2_23.test.predict.tsv"
    vendor_file = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/CBA_video_2_p23.withLabel/CBA_video_2_p23.pureLabel.tsv"
    output_file = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/CBA_video_2_p23.withLabel/toVerify_CBA_video_2_p23.pureLabel.tsv"

    generate_vendor_labels_to_verify(predict_file, vendor_file, output_file)


if __name__ == '__main__':
    main()
