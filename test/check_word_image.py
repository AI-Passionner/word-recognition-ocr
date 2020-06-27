import json
import cv2
import pandas as pd

from src.image_util import ImageUtils
from src.textract_util import TextractUtils

img_response = 'test/test_1/apiResponse.json'
img_file = 'test/test_1/test_1.png'
compare_result = 'test/test_1/compare_result.csv'

data = pd.read_csv(compare_result)
for i, row in data.iterrows():
    if row.match == 0:
        print(row)

with open(img_response, 'r') as r:
    textract_ocr = json.load(r)

word_dict, line_dict = TextractUtils.parse(textract_ocr)
ocr_text = TextractUtils.get_text(word_dict)

image = cv2.imread(img_file)
word_images = TextractUtils.get_word_images(image, word_dict, line_dict)

word_id = '0951e5fb-6772-4106-98ea-0fdd1c964eff'
ImageUtils.show_image(word_images[word_id])
