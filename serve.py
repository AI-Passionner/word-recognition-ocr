import pandas as pd
import cv2
import json
import argparse

from src.predict import RecognitionUtils
from src.textract_util import TextractUtils


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="image file")
ap.add_argument("-r", "--response", type=str, help="textract ocr json file")
ap.add_argument("-o", "--output", type=str, help="path to output model and textract texts")
args = vars(ap.parse_args())

# img_response = 'test/001KLON_Page_1.txt'
# img_file = 'test/001KLON_Page_1.jpg'
# args = {'image': 'test/001KLON_Page_1.jpg',
#         'response': 'test/001KLON_Page_1.txt',
#         'output': './test'
#         }


if __name__ == '__main__':
    result = compare_textract_ocr(args['image'], args['response'], args['output'])
    print(result)
