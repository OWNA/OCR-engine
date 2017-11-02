#!/usr/bin/env python
import argparse
import csv
import glob
import os

import cv2
import logging
import pandas as pd
from pytesseract import pytesseract as pt
from PIL import Image
from tqdm import tqdm

import config
from utils import filter_tess_results, get_scores

logger = logging.getLogger()


def build_tess_annotations():
    annotations = pd.DataFrame(
        columns=['filename', 'cell_type', 'x1', 'x2', 'y1', 'y2', 'content', 'img_width', 'img_height', 'bb_width',
                 'bb_height'])
    # read image files
    for f in tqdm(glob.glob(os.path.join(config.DATA_ROOT, 'dataset/images/*.jpg'))):
        df_bboxes = pd.DataFrame
        bboxes_file = os.path.basename(f)
        tess_ret = pt.run_tesseract(f, bboxes_file, lang=None, boxes=True, config="--oem 2 -l eng tsv")
        bboxes_file += '.tsv'
        if os.path.exists(bboxes_file):
            try:
                # read image
                img = cv2.imread(f, 3 | cv2.IMREAD_IGNORE_ORIENTATION)
                h, w, _ = img.shape
                # read bboxes file and convert coordinates to proper form
                with open(bboxes_file, mode='rb') as bb_file:
                    df_bboxes = pd.read_csv(bb_file, sep='\t', quoting=3,
                                            dtype={'text': str, 'left': int, 'width': int, 'top': int, 'height': int})
                df_bboxes.rename(columns={'text': 'content', 'left': 'x1', 'top': 'y1', 'width': 'bb_width',
                                          'height': 'bb_height'},
                                 inplace=True)
                df_bboxes['x2'] = df_bboxes.x1 + df_bboxes.bb_width
                df_bboxes['y2'] = df_bboxes.y1 + df_bboxes.bb_height
                df_bboxes['filename'] = os.path.basename(f)
                df_bboxes['img_width'] = w
                df_bboxes['img_height'] = h
            finally:
                if os.path.exists(bboxes_file):
                    os.remove(bboxes_file)
                    os.remove(bboxes_file.replace('.tsv', '.box'))
        else:
            logger.warning('Tesseract unable to recognize image: {0}'.format(f))
        annotations = annotations.append(df_bboxes, ignore_index=True)
    annotations['cell_type'] = -1
    # filter bounding boxes
    annotations = filter_tess_results(annotations)
    return annotations


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--tess", action='store_true', default=False,
                    help="Generate annotations with tesseract and save to file")
    ap.add_argument("--benchmark", action='store_true', default=False,
                    help="Benchmark tesseract annotations against ground truth and print metrics")
    args = ap.parse_args()
    if args.tess:
        ann = build_tess_annotations()
        # save annotations
        ann.to_csv('models/tess_annotations.csv')
    if args.benchmark:
        tess_df = pd.read_csv('models/tess_annotations.csv')
        gt = pd.read_csv('models/gt_annotations.csv')
        scores = get_scores(gt, tess_df)
        print(scores.sort('ocr_acc'))
        print('Dataset OCR accuracy:', scores.ocr_acc.mean(), 'Dataset segmentation accuracy:', scores.seg_acc.mean())
