import json
import os
import glob
import pickle
from tqdm import tqdm
import h5py
import argparse
import logging
import requests as r
from base64 import b64encode

allow_text = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '?', 'A', 'Á', 'À', 'Ạ', 'Ấ', 'Ậ', 'B', 'C', 'D', 'E', 'È',
    'Ẹ', 'Ê', 'Ế', 'Ệ', 'Ề', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'Ồ', 'Ộ', 'Ố',
    'Ô', 'P', 'R', 'S', 'T', 'U', 'Ư', 'Ừ', 'Ự', 'Ứ', 'V', 'W', 'Y',
    'Ý', 'Ỳ', 'Ỵ','a', 'á', 'ạ', 'à', 'b', 'c', 'd', 'e', 'è', 'é',
    'ẹ', 'ê', 'ế', 'ề', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'ò', 'ó', 'ọ', 'ô', 'ộ', 'ồ', 'ố', 'p', 'q',
    'r', 's', 't', 'u', 'ụ', 'ú', 'ù', 'v', 'w', 'x', 'y', 'z'
]



def make_image_data_list(images, b64=True):
    """
        image_filenames is a list of filename strings
        Returns a list of dicts formatted as the Vision API
            needs them to be
    """

    def content(context):
        return {
            'image': {'content': context},
            'features': [
                {
                    'type': 'LABEL_DETECTION',
                    'maxResults': 10
                },
                {
                    'type': 'TEXT_DETECTION',
                    'maxResults': 10
                },
                {
                    'type': 'LOGO_DETECTION',
                    'maxResults': 10
                },
                {
                    'type': 'FACE_DETECTION',
                    'maxResults': 10
                },
                {
                    'type': 'LANDMARK_DETECTION',
                    'maxResults': 10
                },
                {
                    'type': 'SAFE_SEARCH_DETECTION',
                    'maxResults': 10
                }
            ]
        }

    img_requests = []
    if not b64:
        for img in images:
            with open(img, 'rb') as f:
                ctxt = b64encode(f.read()).decode()
                img_requests.append(content(ctxt))
    else:
        for img in images:
            img_requests.append(content(img))
    return img_requests


def make_image_data(images, b64=True):
    """
        __docstring__
    """
    img_dict = make_image_data_list(images, b64)
    return json.dumps({"requests": img_dict}).encode()


def request_vision_api(image, api_key, b64=True):
    """
        __docstring___
    """
    response = r.post('https://vision.googleapis.com/v1/images:annotate',
                      data=make_image_data([image], b64),
                      params={'key': api_key},
                      headers={'Content-Type': 'application/json'})
    return response


if __name__ == '__main__':

    API_KEY = 'AIzaSyCUHe6jQ9-27YmSfpVGXK7oVUOOUdm3b88'
    # file save result annotaion
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='./data/images')
    FLAGS = parser.parse_args()
    # list image from 
    images = glob.glob(FLAGS.images + '/*')
    image_annotations = list()
    # logging 
    logging.basicConfig(filename='log_vision.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
    for image in tqdm(images):
        # list contain name_image, words_bboxs, texts, characters_bboxs per image
        if isinstance(image, str):
            # get image's name
            _name = image.split('/')[-1]
            # request google api
            print("[INFO] Get response image: {}".format(_name))
            resp = request_vision_api(image, API_KEY, b64=False)
            # get result
            dict_google_response = json.loads(resp.content)
            # get response
            response = dict_google_response['responses'][0]

            # get text annotaion, get word bboxs -> type []
            text_annotation = response['textAnnotations']
            all_words_bboxs = list()
            all_texts = list()
            
            for idx, each_word in enumerate(text_annotation):
                # storage annotation word per image
                word = list()
                # text_annotation[0] is description full text in image, don't get if not use
                if "locale" in each_word:
                    continue
                else:
                    word_detail = each_word['boundingPoly']
                    text = each_word['description']
                    vertices = word_detail['vertices']  # type list
                    # get four points coords
                    if len(text) > 0:
                        left_top, right_top, right_dow, left_dow = vertices[0], vertices[1], vertices[2], vertices[3]
                        # get value x, y - > [x, y]
                        word.append(list(left_top.values()))
                        word.append(list(right_top.values()))
                        word.append(list(right_dow.values()))
                        word.append(list(left_dow.values()))
                        #
                        all_words_bboxs.append(word)
                        all_texts.append(text)
                    else:
                        continue

            # get full text annotation (characters annotations)
            full_text_ann = response['fullTextAnnotation']
            all_chars_bboxs = list()
            pages = full_text_ann['pages']
            for page in pages:
                for block in page['blocks']:
                    for para in block['paragraphs']:
                        for _word in para['words']:
                            for char in _word['symbols']:  
                                if char['text'] in allow_text:
                                    _char = list()
                                    # get four points char bboxs
                                    left_top_char, right_top_char, right_dow_char, left_dow_char = \
                                                        char['boundingBox']['vertices'][0],\
                                                        char['boundingBox']['vertices'][1],\
                                                        char['boundingBox']['vertices'][2],\
                                                        char['boundingBox']['vertices'][3],\
                                    # get value x, y - > [x, y]
                                    
                                    _char.append(list(left_top_char.values()))
                                    _char.append(list(right_top_char.values()))
                                    _char.append(list(right_dow_char.values()))
                                    _char.append(list(left_dow_char.values()))
                                    # print({char['text']: _char})
                                    #
                                    all_chars_bboxs.append(_char)
                                else:
                                    continue

            image_annotations.append([_name, all_words_bboxs, all_texts, all_chars_bboxs])

        else:
            logging.error("Logging load data", exc_info=True)
            continue
    # print(all_chars_bboxs)
    with open('output/gt.pkl', 'wb') as pkl_file:
        pickle.dump(image_annotations, pkl_file)
