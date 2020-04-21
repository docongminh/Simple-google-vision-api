import pickle
import numpy as np
import cv2
import os
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gt', type=str, default='./')
parser.add_argument('--output', type=str, default='./image_test/all_result')
parser.add_argument('--number_test', type=int, default=-1)
FLAGS = parser.parse_args()

g_truth = FLAGS.gt

def check_exists(path):
	try:
		if not os.path.exists(FLAGS.output):
			os.makedirs(FLAGS.output)
	except Exception as e:
		print(e)
		pass


with open('gt.pkl', 'rb') as pkl_file:
        datas = pickle.load(pkl_file)
        #
path_save_chars = os.path.join(FLAGS.output, "chars")
path_save_words = os.path.join(FLAGS.output, "words")
path_save_text = os.path.join(FLAGS.output, "text")
#
check_exists(path_save_chars)
check_exists(path_save_words)
check_exists(path_save_text)


path = 'data/images'
imgs = datas[:FLAGS.number_test]

for data in tqdm.tqdm(imgs):
	image_name = data[0]
	word_bboxs = data[1]
	texts = data[2]
	chars_bboxs = data[3]

	# print("all texts: ", texts)

	img = cv2.imread(os.path.join(path, image_name))
	_n = image_name.split('.')[0]
	with open("image_test/all_result/text/{}.txt".format(_n), 'w') as f:
		for word_text in texts:
			f.write(word_text)
			f.write("\n")

	word_image = img.copy()
	for bb in word_bboxs:
		top, _, down, _ = bb
		x, y, xx, yy = int(top[0]), int(top[1]), int(down[0]), int(down[1])
		word_rect = cv2.rectangle(word_image, (x, y), (xx, yy), (255, 0, 255), 2)
	cv2.imwrite('{}/words/word_rect_{}.jpg'.format(FLAGS.output, _n), word_rect)
	#

	char_image = img.copy()
	for char in chars_bboxs:
		left_above, _, right_bottom, _ = char
		# print(">> char: ", char)
		rect = cv2.rectangle(char_image,(int(left_above[0]), int(left_above[1])),\
						(int(right_bottom[0]), int(right_bottom[1])), (255,0,255), 2)
	cv2.imwrite('{}/chars/char_rect_{}.jpg'.format(FLAGS.output, _n), rect)