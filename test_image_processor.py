import cv2
import os

from pipeline import *

images = os.listdir('test_images')

output_folder = 'test_images_output'

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

for image in images:
    img = cv2.imread('test_images/' + image)
    pipeline_output = pipeline(img)
    cv2.imwrite(os.path.join(output_folder, image), pipeline_output)
