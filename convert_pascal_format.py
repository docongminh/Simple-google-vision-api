import pickle
import numpy as np
import cv2
import os
import tqdm
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='./dataset/result')
parser.add_argument('--input', type=str, default='./dataset/data')
parser.add_argument('--gt_path', type=str, default='./gt.pkl')
parser.add_argument('--number_test', type=int, default=-1)
FLAGS = parser.parse_args()


def check_exists(path):
	try:
		if not os.path.exists(path):
			os.makedirs(path)
	except Exception as e:
		raise e

with open(FLAGS.gt_path, 'rb') as pkl_file:
        datas = pickle.load(pkl_file)
        #
path_save_chars = os.path.join(FLAGS.output, "chars")
path_save_words = os.path.join(FLAGS.output, "words")
path_save_text = os.path.join(FLAGS.output, "text")
path_save_ann = os.path.join(FLAGS.output, "annotation")
#
check_exists(path_save_chars)
check_exists(path_save_words)
check_exists(path_save_text)
check_exists(path_save_ann)
# print(datas)
print(len(datas))
path = FLAGS.input
imgs = datas[:FLAGS.number_test]
#
# logging 
logging.basicConfig(filename='log_annotation.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
for data in tqdm.tqdm(datas):
	try:
		# print(data)
		image_name = data[0]
		word_bboxs = data[1]
		texts = data[2]
		chars_bboxs = data[3]

		# print("all texts: ", texts)
		# print(">>>name: ", image_name)
		# print(os.path.join(path, image_name))
		img = cv2.imread(os.path.join(path, image_name))
		# print("image: ", img)
		# _n = image_name.split('.')[0]
		_n = os.path.splitext(image_name)[0]
		with open("{}/{}.txt".format(path_save_ann, _n), 'w') as f:
			for idx, bb in enumerate(word_bboxs):
				top, _, down, _ = bb
				x, y, xx, yy = int(top[0]), int(top[1]), int(down[0]), int(down[1])
				f.write(texts[idx])
				f.write("\t")
				f.write(str(x))
				f.write("\t")
				f.write(str(y))
				f.write("\t")
				f.write(str(xx))
				f.write("\t")
				f.write(str(yy))
				f.write("\n")

		word_image = img.copy()
		for idx, bb in enumerate(word_bboxs):
			top, _, down, _ = bb
			x, y, xx, yy = int(top[0]), int(top[1]), int(down[0]), int(down[1])
			word_rect = cv2.rectangle(word_image, (x, y), (xx, yy), (255, 0, 255), 2)
		cv2.imwrite('{}/word_rect_{}.jpg'.format(path_save_words, _n), word_rect)
		#

		char_image = img.copy()
		for char in chars_bboxs:
			left_above, _, right_bottom, _ = char
			# print(">> char: ", char)
			rect = cv2.rectangle(char_image,(int(left_above[0]), int(left_above[1])),\
							(int(right_bottom[0]), int(right_bottom[1])), (255,0,255), 2)
		cv2.imwrite('{}/char_rect_{}.jpg'.format(path_save_chars, _n), rect)
	except Exception as e:
		logging.error("Logging load data", exc_info=True)
		continue