import pickle
import numpy as np
import cv2
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gt', type=str, default='./')
parser.add_argument('--output', type=str, default='./image_test')
parser.add_argument('--number_test', type=int, default=2)
FLAGS = parser.parse_args()

g_truth = FLAGS.gt

with open('gt.pkl', 'rb') as pkl_file:
        datas = pickle.load(pkl_file)
        #
path = 'images'
imgs = data[:FLAGS.number_test]
for data in imgs:
	image_name = data[0]
	word_bboxs = data[1]
	texts = data[2]
	chars_bboxs = data[3]
	#
	print(">>>>> Print Words bboxs: ")
	print(word_bboxs)
	np_word = np.asarray(word_bboxs)
	print(">> shape words bboxs: ", np_word.shape)
	print()
	print("all texts: ", texts)
	print()
	print(">>>> print Chars bboxs")
	print(chars_bboxs)
	np_chars = np.asarray(chars_bboxs)
	print(">> shape chars bboxs: ", np_chars.shape)
	#
	img = cv2.imread(os.path.join(path, image_name))
	print()
	print(">>> Draw words bboxs: ", image_name)
	_n = image_name.split('.')[0]
	word_image = img.copy()
	for bb in word_bboxs:
		top, _, down, _ = bb
		x, y, xx, yy = int(top[0]), int(top[1]), int(down[0]), int(down[1])
		word_rect = cv2.rectangle(word_image, (x, y), (xx, yy), (255, 0, 255), 2)
	cv2.imwrite('{}/word_rect_{}.jpg'.format(FLAGS.output, _n), word_rect)
	#
	print()
	print(">>> Draw chars bboxs: ", image_name)

	char_image = img.copy()
	for char in chars_bboxs:
		left_above, _, right_bottom, _ = char
		print(">> char: ", char)
		rect = cv2.rectangle(char_image,(int(left_above[0]), int(left_above[1])),\
						(int(right_bottom[0]), int(right_bottom[1])), (255,0,255), 2)
	cv2.imwrite('{}/char_rect_{}.jpg'.format(FLAGS.output, _n), rect)