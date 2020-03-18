import cv2
import json
from qd.process_tsv import ocr_tsv

def run_parallel_OCR_on_TSV(in_tsv, out_tsv):
    #urls = ['http://vigdgx01:5001/vision/v2.0/read/core/Analyze']
    urls = ['http://52.151.6.35:5000/formrecognizer/v2.0-preview/readLayout/analyze']

    ocr_tsv(in_tsv, out_tsv, urls)

def ocr_testing_images():
    in_tsv = '/home/yaowe/gavin/productRecog/data/Mars_data_10-18_train0.1/MarsData.det.test.images.patches.no_product.tsv'
    out_tsv = '/home/yaowe/gavin/productRecog/data/Mars_data_10-18_train0.1/MarsData.det.test.images.patches.no_product.OCR.tsv'
    run_parallel_OCR_on_TSV(in_tsv, out_tsv)

def ocr_testing_images_test():
    in_tsv = '/home/yaowe/gavin/productRecog/data/Mars_data_10-18_train0.1/testOCRtest.tsv'
    out_tsv = '/home/yaowe/gavin/productRecog/data/Mars_data_10-18_train0.1/testOCRtest.OCR.tsv'
    run_parallel_OCR_on_TSV(in_tsv, out_tsv)

def ocr_canonical_images():
    in_tsv = '/home/yaowe/gavin/productRecog/data/canonical_thumbnail_tsv/canonical_thumbnail.tsv'
    out_tsv = '/home/yaowe/gavin/productRecog/data/canonical_thumbnail_tsv/canonical_thumbnail.OCR.tsv'
    run_parallel_OCR_on_TSV(in_tsv, out_tsv)

if __name__ == '__main__':
    #ocr_testing_images()
    #ocr_testing_images_test()
    ocr_canonical_images()