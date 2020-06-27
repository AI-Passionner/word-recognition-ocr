import numpy as np

from src.image_util import ImageUtils


class TextractUtils:

    @staticmethod
    def parse(textract_ocr):
        """Parses the Textract JSON Object and returns the word map and line map"""
        blocks = textract_ocr['Blocks']
        word_dict = {}
        line_dict = {}
        for block in blocks:
            block_id = block['Id']
            if block['BlockType'] == "WORD":
                bbox = block['Geometry']['BoundingBox']
                x, y, w, h = bbox['Left'], bbox['Top'], bbox['Width'], bbox['Height']
                word_dict[block_id] = {'ids': [block_id],
                                       'Text': block['Text'],
                                       'left': x,
                                       'top': y,
                                       'right': x + w,
                                       'bottom': y + h,
                                       'height': h,
                                       'width': w,
                                       'bbox': [x, y, x + w, y + h],
                                       'confidence': block['Confidence']
                                       }

            elif block['BlockType'] == "LINE":
                if 'Relationships' in block.keys():
                    for relationship in block['Relationships']:
                        if relationship['Type'] == 'CHILD':
                            ids = relationship['Ids']

                    bbox = block['Geometry']['BoundingBox']
                    x, y, w, h = bbox['Left'], bbox['Top'], bbox['Width'], bbox['Height']
                    line_dict[block_id] = {'ids': ids,
                                           'Text': block['Text'],
                                           'left': x,
                                           'top': y,
                                           'right': x + w,
                                           'bottom': y + h,
                                           'height': h,
                                           'width': w,
                                           'bbox': [x, y, x + w, y + h],
                                           'confidence': block['Confidence']
                                           }

        return word_dict, line_dict

    @staticmethod
    def get_text(word_dict):
        return ' '.join(v['Text'] for k, v in word_dict.items())

    @staticmethod
    def get_word_images(raw_img, word_dict, line_dict):
        """
        Parameters
        ----------
        raw_img : numpy array of image
        word_dict : python dictionary, storing word-level OCR
        line_dict : python dictionary, storing line-level OCR
        Returns
        -------
        a dictionary,  having {word_id: word_image, ...}

        """

        img_thresh = ImageUtils.image_for_extraction(raw_img)
        # remove lines not good for highlighted words
        # img_thresh = ImageUtils.remove_lines(img_thresh, hori_size=120, vert_size=100 )
        img_h, img_w = img_thresh.shape[:2]

        h_w_ratio = [v['height'] / v['width'] for k, v in word_dict.items()]
        if np.mean(h_w_ratio) > 1:  # rotated images
            raise Exception('Rotated images are not supported')

        word_imgs = {}
        for k, v in line_dict.items():
            bbox = v['bbox']
            l, t, r, b = ImageUtils.reverse_bbox(img_h, img_w, bbox)
            line_img = img_thresh[t:b, l:r]
            line_img = ImageUtils.crop_image(line_img, axis=2)
            height_in_line = line_img.shape[0]
            ids = v['ids']
            ids = sorted(ids, key=lambda x: word_dict[x]['left'])
            for i in ids:
                bbox0 = word_dict[i]['bbox']
                l0, t0, r0, b0 = ImageUtils.reverse_bbox(img_h, img_w, bbox0)
                word_img = img_thresh[t0:b0, l0:r0]
                word_img = ImageUtils.crop_image(word_img, 2)
                size = word_img.shape[:2]
                half_h = max(0, int((height_in_line - size[0]) / 2))
                word_img = np.pad(word_img, ((half_h, half_h), (0, 0)), 'constant', constant_values=0)
                word_imgs[i] = word_img

        return word_imgs


