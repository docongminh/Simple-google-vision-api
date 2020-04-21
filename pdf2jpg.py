from pdf2image import convert_from_path
import os
import random

path_pdf = 'data/pdf'
path_img = 'data/images'
if not os.path.exists(path_img):
	os.mkdir(path_img)

pdfs = os.listdir(path_pdf)
random.shuffle(pdfs)
for idx, pdf in enumerate(pdfs):
	file = os.path.join(path_pdf, pdf)
	print("file: ", file)
	name = pdf.split(".")[0][:5]
	pages = convert_from_path(file, 500)
	print("pages: ", pages)
	for idx_page, page in enumerate(pages):
		page.save(path_img+'/name_{}_{}.jpg'.format(idx, idx_page), 'JPEG')
