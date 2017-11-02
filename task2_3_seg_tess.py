#!/usr/bin/env python
import argparse
import csv
import glob
import os
import numpy as np
import cv2
import logging
import pandas as pd
from pytesseract import pytesseract as pt
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt

import config
from utils import filter_tess_results, get_scores

logger = logging.getLogger()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", action='store_true', default=False,
                    help="Generate annotations with tesseract and save to file")
    ap.add_argument("--benchmark", action='store_true', default=False,
                    help="Benchmark tesseract annotations against ground truth and print metrics")
    args = ap.parse_args()

    if args.ocr:
        annotations = pd.DataFrame(
            columns=['filename', 'cell_type', 'x1', 'x2', 'y1', 'y2', 'content', 'img_width', 'img_height', 'bb_width',
                     'bb_height'])

        for filename in tqdm(glob.glob(os.path.join(config.DATA_ROOT, 'dataset/images/*.jpg')), 'Processing files'):
            # load annotations to get roi, as roi detection is out of scope of the task
            gt_df = pd.read_csv('models/gt_annotations.csv')
            gt_data = gt_df[gt_df.filename == os.path.basename(filename)]
            roi = gt_data.x1.min() - 20, gt_data.y1.min() - 20, gt_data.x2.max() + 20, gt_data.y2.max() + 20

            img_pipeline = []
            src_img = cv2.imread(filename, 3 | cv2.IMREAD_IGNORE_ORIENTATION)
            img_pipeline.append(('src', src_img))
            img_pipeline.append(('crop', img_pipeline[-1][1][roi[1]:roi[3], roi[0]:roi[2]]))
            img_pipeline.append(('grayscale', cv2.cvtColor(img_pipeline[-1][1], cv2.COLOR_BGR2GRAY)))
            img_pipeline.append(('gauss_blur', cv2.GaussianBlur(img_pipeline[-1][1], (5, 5), 0)))
            thr, img = cv2.threshold(img_pipeline[-1][1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img_pipeline.append(('otsu_thresh', img))
            dil_kernel = np.ones((10, 10), np.uint8)
            img_pipeline.append(
                ('dilate', cv2.bitwise_not(cv2.dilate(cv2.bitwise_not(img_pipeline[-1][1]), kernel=dil_kernel))))

            mser = cv2.MSER_create()
            reg_detect = img_pipeline[-1][1].copy()
            regions = mser.detectRegions(reg_detect)
            reg_detect = cv2.cvtColor(reg_detect, cv2.COLOR_GRAY2BGR)
            hulls = [cv2.rectangle(reg_detect, (p[0], p[1]), (p[0] + p[2], p[1] + p[3]), (0, 255, 0)) for p in regions[1]]
            img_pipeline.append(('reg', reg_detect))

            # convert bboxes coordinates
            bboxes = [(p[0] + roi[0], p[1] + roi[1], p[0] + p[2] + roi[0], p[1] + p[3] + roi[1]) for p in regions[1]]

            # draw on original image
            src_copy = img_pipeline[0][1].copy()
            [cv2.rectangle(src_copy, (b[0], b[1]), (b[2], b[3]), (0, 255, 0)) for b in bboxes]
            img_pipeline.append(('orig_bboxes', src_copy))

            # recognize bboxes with tess
            for b in tqdm(bboxes, 'Processing bboxes'):
                img = Image.fromarray(src_img[b[1]:b[3], b[0]:b[2]])
                img = img.rotate(-90, expand=True)
                text = pt.image_to_string(img, config="--oem 2 --psm 8 -l eng")
                # img.show()
                annotations = annotations.append(
                    {'filename': os.path.basename(filename), 'cell_type': -1, 'x1': b[0], 'x2': b[2], 'y1': b[1], 'y2': b[3], 'content': text,
                     'img_width': src_img.shape[1], 'img_height': src_img.shape[0], 'bb_width': b[2] - b[0],
                     'bb_height': b[3] - b[1]}, ignore_index=True)

        annotations['conf'] = 100
        # intentionally taking shortcut of not building proper chains and detecting orientation here
        annotations = annotations.sort(columns=['y1', 'x1'], ascending=[0, 1])
        annotations = filter_tess_results(annotations)
        annotations.to_csv('models/seg_tess_annotations.csv')

        # img_pipeline.reverse()
        # for desc, img in img_pipeline:
        #     img = cv2.resize(img, (1080, 800))
        #     cv2.imshow(desc, img)
        #     cv2.waitKey(0)
        #     # plt.imshow(img[::-1])
        #     pass

    if args.benchmark:
        tess_df = pd.read_csv('models/seg_tess_annotations.csv')
        gt = pd.read_csv('models/gt_annotations.csv')
        scores = get_scores(gt, tess_df)
        print(scores.sort('ocr_acc'))
        print('Dataset OCR accuracy:', scores.ocr_acc.mean(), 'Dataset segmentation accuracy:', scores.seg_acc.mean())

