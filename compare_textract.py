import pandas as pd
import cv2
import json
import argparse

from src.predict import RecognitionUtils
from src.textract_util import TextractUtils


def compare_textract_ocr(img_file, img_response, output_path):

    with open(img_response, 'r') as r:
        textract_ocr = json.load(r)

    img = cv2.imread(img_file)
    word_dict, line_dict = TextractUtils.parse(textract_ocr)
    Text = TextractUtils.get_text(word_dict)

    word_images = TextractUtils.get_word_images_from_textract(img, word_dict, line_dict)
    recog_words = RecognitionUtils.recognize(word_images)
    text = ' '.join(recog_words[k]['Text'] for k in recog_words)

    data = []
    tp = 0
    for k, v in recog_words.items():
        p_txt = v['Text']
        t_txt = word_dict[k]['Text']
        match = 1 if p_txt == t_txt else 0
        tp += match
        data.append([k, t_txt, p_txt, round(v['Confidence'], 4), match])

    accuracy = tp / len(data)

    data = pd.DataFrame(data, columns=['word_id', 'textract_text', 'pred_text', 'pred_conf', 'match'])
    data.to_csv(output_path + '/' + 'output.csv', index=False)

    return {'Textract_Text': Text, 'In_house_Text': text, 'Accuracy': accuracy}


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
