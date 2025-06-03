from reader import read_data
import numpy as np 
from PIL import Image
'''
This script saves NUM_EXAMPLES random digits from train dataset in current folder
'''
NUM_EXAMPLES = 10
###
x_train, y_train, x_test, y_test = read_data()

def save_img(idx):
    arr = x_train[idx]
    lbl = y_train[idx]
    img = Image.fromarray(arr)
    img_name = str(lbl) + '_' + str(idx) + '.png'
    img.save(img_name)

for i in range(NUM_EXAMPLES):
    r = np.random.randint(60000)
    save_img(r)

print('Done')
