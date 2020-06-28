import cv2
import numpy as np
from uuid import uuid4
from itertools import groupby

from src.image_util import ImageUtils


class AnalyzeDocument:
    """ Challenging parts:
    1. need to extract embedded word images. Using contour might be a good try.
    2. need to find the a dynamic/better for the word segmentation.
    """

    @staticmethod
    def is_vertical(bbox1, bbox2):
        """
        Checks if bounding boxes are vertically aligned
        box: [left, top, right, bottom]
        :param bbox1: Bounding box
        :param bbox2: bounding box

        :return True is bounding boxes are vertically aligned else False
        """
        l1, t1, r1, b1 = bbox1
        l2, t2, r2, b2 = bbox2
        return (l1 <= r2 <= r1) or (l2 < r1 <= r2)

    @staticmethod
    def is_horizontal(bbox1, bbox2):
        """
        Checks if bounding boxes are horizontally aligned
        box: [left, top, right, bottom]
        :param bbox1: Bounding box
        :param bbox2: bounding box
        :return True is bounding boxes are horizontally aligned else False
        """
        l1, t1, r1, b1 = bbox1
        l2, t2, r2, b2 = bbox2
        return (t2 <= t1 < b2) or (t1 <= t2 < b1)

    @staticmethod
    def cluster_bboxes_in_line(bboxes, ids):
        """
        Clusters bounding boxes in the line, recursive solution.
        (bounding_boxes and ids are paired)
        :param bboxes: bounding boxes representing line.
        :param ids: ids of bounding boxes
        :return: clustered bounding boxes in line.
        """
        n = len(ids)
        if n == 0:
            return []
        elif n == 1:
            return [ids]
        else:
            ids_ex = []
            bboxs_ex = []
            ids_inline = [ids[0]]
            p_0 = bboxes[0]
            for id_j, p_j in zip(ids[1:], bboxes[1:]):
                if AnalyzeDocument.is_horizontal(p_j, p_0):
                    ids_inline.append(id_j)
                    p_0 = p_j

                else:
                    ids_ex.append(id_j)
                    bboxs_ex.append(p_j)

            return [ids_inline] + AnalyzeDocument.cluster_bboxes_in_line(bboxs_ex, ids_ex)

    @staticmethod
    def detect_text_contour(thresh, kernel=(2, 6)):
        kernel = np.ones(kernel, np.uint8)  # making the second element bigger to connect words
        # use closing morph operation but fewer iterations than the letter then erode to narrow the image
        temp_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        kernel2 = np.ones((2, 2), np.uint8)
        dilate_img = cv2.dilate(temp_img, kernel2, iterations=2)

        # find contours of texts
        contours, hierarchy = cv2.findContours(dilate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = sorted(contours, key=lambda c: min(min(c[:, :, 0])))

        return contours

    @staticmethod
    def get_t_threshold(x, q=0.95, min_threshold=6):
        from scipy.stats import t
        m = np.mean(x)
        s = np.std(x)
        df = len(x) - 1
        score = t.ppf(q, 10)
        threshold = np.ceil(score*s + m)
        return max(threshold, min_threshold)

    @staticmethod
    def get_mad_threshold(x, q=0.95, min_threshold=6):
        x = np.array(x)
        m = np.median(x)
        abs_m = np.median(abs(x - m))
        mad = 1.4826 * abs_m
        return mad

    @staticmethod
    def find_lines(bw_image, orientation):
        if orientation == 'horizontal':
            proj = np.mean(bw_image, axis=1)
        elif orientation == 'vertical':
            proj = np.mean(bw_image, axis=0)

        hist = [0 if x <= 0 else 1 for x in proj]
        lines = []
        counts = []
        index = 0
        for k, g in groupby(hist):
            n = len(list(g))
            if k == 0:
                lines.append(index + 1)
                counts.append(n)

            index += n

        return counts, lines

    @staticmethod
    def sort_block_contours(block_contours):
        num_cluster = len(block_contours)
        block_boxes = [cv2.boundingRect(c) for c in block_contours]
        block_boxes = [[x, y, x + w, y + h] for x, y, w, h in block_boxes]
        clusters_in_line = AnalyzeDocument.cluster_bboxes_in_line(block_boxes, list(range(num_cluster)))
        clusters_in_line = sorted(clusters_in_line, key=lambda x: block_boxes[x[0]][1], reverse=False)
        clusters_in_line = [sorted(line, key=lambda x: block_boxes[x][0], reverse=False) for line in clusters_in_line]
        sorted_block_bboxs = [block_boxes[i] for line in clusters_in_line for i in line]

        return sorted_block_bboxs

    @staticmethod
    def extract_blocks(img_for_ext):
        block_contours = AnalyzeDocument.detect_text_contour(img_for_ext, kernel=(25, 25))
        sorted_block_bboxs = AnalyzeDocument.sort_block_contours(block_contours)

        return sorted_block_bboxs

    @staticmethod
    def extract_word_images(img_for_ext, sorted_block_bboxs):
        img_h, img_w = img_for_ext.shape[:2]
        word_images = {}
        for bbox in sorted_block_bboxs:
            l, t, r, b = bbox
            l, t, r, b = max(0, l - 4), max(0, t - 4), min(img_w, r + 2), min(img_h, b + 2)
            block_image = ImageUtils.crop_image(img_for_ext[t:b, l:r], axis=0)
            block_h = block_image.shape[0]
            hori_counts, hori_lines = AnalyzeDocument.find_lines(block_image, 'horizontal')
            line_images = []
            if hori_lines:
                t = 0
                for b in hori_lines:
                    line_images.append(block_image[t:b, :])
                    t = b

                line_images.append(block_image[t:, :])
            else:
                line_images.append(block_image)

            for line_image in line_images:
                line_image = ImageUtils.crop_image(line_image, axis=2)
                vert_counts, vert_lines = AnalyzeDocument.find_lines(line_image, 'vertical')
                if vert_lines:
                    kernel = np.ones((4, int(2*np.median(vert_counts))), np.uint8)
                    # use closing morph operation but fewer iterations than the letter then erode to narrow the image
                    temp_img = cv2.morphologyEx(line_image, cv2.MORPH_CLOSE, kernel, iterations=2)
                    kernel2 = np.ones((2, 2), np.uint8)
                    dilate_img = cv2.dilate(temp_img, kernel2, iterations=2)
                    vert_counts, vert_lines = AnalyzeDocument.find_lines(dilate_img, 'vertical')
                    # t_threhold = get_t_threshold(vert_counts, q=0.95, min_threshold=0)
                    # vert_lines = [l for l, c in zip(vert_lines, vert_counts) if c >= t_threhold]
                    if vert_lines:
                        l=0
                        for r in vert_lines:
                            iid = str(uuid4())
                            word_image = ImageUtils.crop_image(line_image[:, l:r], axis=1)
                            word_images[iid] = word_image
                            l = r

                        word_image = ImageUtils.crop_image(line_image[:, l:], axis=1)
                        iid = str(uuid4())
                        word_images[iid] = word_image
                    else:
                        iid = str(uuid4())
                        word_image = ImageUtils.crop_image(line_image, axis=1)
                        word_images[iid] = word_image

                else:
                    iid = str(uuid4())
                    word_image = ImageUtils.crop_image(line_image, axis=1)
                    word_images[iid] = word_image

        return word_images




