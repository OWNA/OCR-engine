import os
import numpy as np
import cv2
import pandas as pd
from pylev import levenshtein
import config


def filter_tess_results(df_tess):
    # filter by confidence
    df_tess = df_tess[df_tess.conf > 0]
    # filter na's
    df_tess.dropna(inplace=True)
    # filter empty characters
    df_tess = df_tess[np.logical_and(df_tess.content != '', df_tess.content != ' ')]
    # filter invalid bounding boxes
    df_tess = df_tess[np.logical_and(df_tess.bb_width > 0, df_tess.bb_height > 0)]
    df_tess = df_tess[df_tess.x2 < df_tess.img_width]
    df_tess = df_tess[df_tess.x2 > 0]
    df_tess = df_tess[df_tess.y2 < df_tess.img_height]
    df_tess = df_tess[df_tess.y2 > 0]
    return df_tess


def sort_df(df, column_idx, key):
    '''Takes dataframe, column index and custom function for sorting,
    returns dataframe sorted by this column using this function'''

    col = df.ix[:, column_idx]
    temp = np.array(col.values.tolist())
    order = sorted(range(len(temp)), key=lambda j: key(temp[j]))
    return df.ix[order]


def get_scores(gt_df, ocr_df):
    """
    Calculates accuracy score for text segmentation and text recognition
    :param gt_df: ground truth data
    :param ocr_df: ocr results
    :return:
    """
    roi_shift = 10
    bbox_shift = 10
    global_seg_acc = 0
    global_ocr_acc = 0
    file_count = 0
    df_metrics = pd.DataFrame(
        columns=['filename', 'seg_acc', 'ocr_acc', 'bbox_count', 'bbox_count_gt', 'bbox_matches', 'img_width',
                 'img_height'])
    for filename, gt in gt_df.groupby('filename'):
        ocr = ocr_df[ocr_df.filename == filename]
        total_seg_area = 0
        bbox_matches = 0
        total_ocr_acc = 0
        seg_acc = 0
        total_wrong_seg_area = 0
        if ocr.shape[0] > 0:
            # TODO: no bboxes for file at all situation doesn't addressed by metrics
            file_count += 1
            # infer region of interest from gt data
            roi = (gt.x1.min() - roi_shift, gt.y1.min() - roi_shift, gt.x2.max() + roi_shift, gt.y2.max() + roi_shift)
            # drop other boxes
            ocr = ocr[(ocr.x1 > roi[0]) & (ocr.y1 > roi[1]) & (ocr.x2 < roi[2]) & (ocr.y2 < roi[3])]
            total_seg_area = (ocr.bb_width * ocr.bb_height).sum()
            # GT data have no info on text flow direction, so theoretically we don't know in which order to assemble bboxes
            # let's assume that boxes are in normal reading order
            for i, gt_row in gt.iterrows():
                ocr_bboxes = ocr[(ocr.x1 > gt_row.x1 - bbox_shift) & (ocr.y1 > gt_row.y1 - bbox_shift) &
                                 (ocr.x2 < gt_row.x2 + bbox_shift) & (
                                     ocr.y2 < gt_row.y2 + bbox_shift)]  # type:pd.DataFrame
                bbox_matches += len(ocr_bboxes)
                ocr_content = ' '.join(' '.join(ocr_bboxes.content).split(' '))
                gt_content = ' '.join(gt_row.content.split(' '))
                if len(ocr_content) == 0:
                    ocr_acc = 0
                else:
                    # levenshtein-based accuracy metric
                    ocr_acc = len(gt_content) * max(0,
                                                    1 - levenshtein(ocr_content, gt_content) / float(len(gt_content)))
                    # drop all used bboxes
                    ocr.drop(ocr_bboxes.index)
                # estimation of seg correctness is not trivial, plus gt data isn't segmented correctly, very long spaces should be 2 bboxes, not one, I think
                # so let's assume seg correctness is proportional to the number of symbols extracted compared to gt
                wrong_seg_area = (gt_row.bb_width * gt_row.bb_height) * abs(
                    len(ocr_content) - len(gt_content)) / float(len(gt_content))
                total_wrong_seg_area += wrong_seg_area
                total_ocr_acc += ocr_acc
            # the rest bboxes are false positives
            total_ocr_acc /= gt.content.str.len().sum()
            seg_acc = 0 if total_seg_area == 0 else max(0,
                                                        (total_seg_area - total_wrong_seg_area) / float(total_seg_area))
            global_seg_acc += seg_acc
            global_ocr_acc += total_ocr_acc
            print('Image:', filename, 'Segmentation accuracy:', seg_acc, 'OCR accuracy:', total_ocr_acc)
        df_metrics = df_metrics.append(
            {'filename': filename, 'seg_acc': seg_acc, 'ocr_acc': total_ocr_acc, 'bbox_count': len(ocr),
             'bbox_count_gt': len(gt), 'bbox_matches': bbox_matches, 'img_width': gt.img_width.iloc[0],
             'img_height': gt.img_height.iloc[0]}, ignore_index=True)
    return df_metrics

def convert_annotations():
    """
    Convert annotations to a more convenient format
    :return:
    """
    annotations = pd.read_csv(os.path.join(config.DATA_ROOT, 'dataset/elements.dat'),
                              dtype={'content': unicode, 'l': int, 'r': int, 't': int, 'b': int, 'cell_type': int},
                              index_col=None)
    annotations.rename(columns={'l': 'x1', 'r': 'x2', 't': 'y1', 'b': 'y2'}, inplace=True)

    # add file dimensions
    def get_size(filename):
        file_path = os.path.join(config.DATA_ROOT, 'dataset/images', filename)
        if os.path.exists(file_path):
            img = cv2.imread(file_path)
            return img.shape[:2][::-1]
        return np.nan, np.nan

    annotations.filename = annotations.filename.str.replace('images/', '')
    filenames = annotations.filename.unique()
    filedims = map(get_size, filenames)
    filedims_df = pd.DataFrame(zip(filenames, [w[0] for w in filedims], [h[1] for h in filedims]),
                               columns=['filename', 'img_width', 'img_height'])
    annotations = annotations.merge(filedims_df, on=['filename'])
    annotations.dropna(inplace=True)
    annotations.img_width, annotations.img_height = annotations.img_width.astype(
        np.int), annotations.img_height.astype(
        np.int)
    # add corrective rotation gt col
    annotations.rotation = 0
    annotations['bb_width'] = np.abs(annotations.x1 - annotations.x2)
    annotations['bb_height'] = np.abs(annotations.y1 - annotations.y2)

    # save
    annotations.to_csv('gt_annotations.csv', index=None)

    # display bboxes
    # for g_key, g in annotations.groupby('filename'):
    #     img = cv2.imread(os.path.join(config.DATA_ROOT, 'dataset', g_key), 3 | cv2.IMREAD_IGNORE_ORIENTATION)
    #     for i, b in g.iterrows():
    #         img = cv2.rectangle(img, (b.x1, b.y1), (b.x2, b.y2), (255, 0, 0), 2)
    #     img = cv2.resize(img, (1920, 1080))
    #     cv2.imshow('', img)

if __name__ == '__main__':
    convert_annotations()
