import cv2
import argparse

from src.predict import RecognitionUtils
from src.analyze_document import AnalyzeDocument
from src.image_util import ImageUtils


def main(image_file, output_path):
    image = cv2.imread(image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # remove lines for better txt detection
    bw_img = ImageUtils.remove_lines(gray, 120, 100)  # white space and black text
    cv2.imwrite(output_path + '/' + 'line_removed.png', bw_img)

    ret, img_for_ext = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(output_path + '/' + 'img_for_extr.png', img_for_ext)

    block_bboxs = AnalyzeDocument.extract_blocks(img_for_ext)
    img_h, img_w = img_for_ext.shape[:2]
    masked_img = image.copy()
    for bbox in block_bboxs:
        l, t, r, b = bbox
        l, t, r, b = max(0, l - 4), max(0, t - 4), min(img_w, r + 2), min(img_h, b + 2)
        cv2.rectangle(masked_img, (l, t), (r, b), (0, 0, 255), 2)
    cv2.imwrite(output_path + '/' + 'block_img.png', masked_img)

    word_images = AnalyzeDocument.extract_word_images(img_for_ext, block_bboxs)
    recog_words = RecognitionUtils.recognize(word_images)
    text = ' '.join(recog_words[k]['Text'] for k in recog_words)
    print(text)

    with open(output_path + '/' + 'output.txt', 'w') as f:
        f.write(text)

    return


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="image file")
ap.add_argument("-o", "--output", type=str, help="path to output model and textract texts")
args = vars(ap.parse_args())

# args = {
#         'image': './test/test_1/test_1.png',
#         'output': './test/test_1'
#         }


if __name__ == '__main__':
    main(args['image'], args['output'])

