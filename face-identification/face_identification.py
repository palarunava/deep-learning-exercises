import json
import os
import urllib
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

#%%
path = './face-detection-in-images'
data_path = os.path.join(os.path.abspath(path), 'face_detection.json')
data =  open(data_path).readlines()

#%%
images_path = os.path.join(os.path.abspath(path), 'images')
if os.path.exists(images_path):
    shutil.rmtree(images_path)
os.mkdir(images_path)

#%%
def download_image (image_url):
    image_name = image_url.split('/')[-1]
    image_path = os.path.join(images_path, image_name)
    urllib.request.urlretrieve(image_url, image_path)
    return image_name
    
#%%
image_names = [download_image(json.loads(item)['content']) for item in data]
image_size = [(json.loads(item)['annotation'][0]['imageWidth'], json.loads(item)['annotation'][0]['imageHeight']) for item in data]

#%%
def recalculate_bounding_box (bounding_box, image_width, image_height):
    top_left = (bounding_box[0][0] * image_width, bounding_box[0][1] * image_height)
    width = (bounding_box[1][0] - bounding_box[0][0]) * image_width
    height = (bounding_box[1][1] - bounding_box[0][1]) * image_height
    center = (top_left[0] + (width / 2), top_left[1] + (height / 2))
    return center, width, height;    

#%%
images = pd.DataFrame(columns=('URL', 'NAME', 'WIDTH', 'HEIGHT', 'BOUNDING_BOXES'))
for image_data in data:
    image_url = json.loads(image_data)['content']
    image_name = image_url.split('/')[-1]
    #image_name = download_image(image_url)
    image_width = json.loads(image_data)['annotation'][0]['imageWidth']
    image_height = json.loads(image_data)['annotation'][0]['imageHeight']
    bounding_boxes = [[(item['points'][0]['x'], item['points'][0]['y']), (item['points'][1]['x'], item['points'][1]['y'])] for item in json.loads(image_data)['annotation']]
    bounding_boxes = [recalculate_bounding_box(item, image_width, image_height) for item in bounding_boxes]
    images = images.append(pd.Series([image_url, image_name, image_width, image_height, bounding_boxes], index=images.columns), ignore_index=True)

#%%
def display_image (sample_image, scale):
    if scale is None:
        test_image = image.load_img(os.path.join(images_path, sample_image['NAME']))
    else:
        test_image = image.load_img(os.path.join(images_path, sample_image['NAME']), target_size = scale)
    plt.figure(figsize=(10,10))
    plt.imshow(test_image)    
    ax = plt.gca()
    ax.xaxis.set_ticks([i for i in range(0, test_image.size[0], 19)])
    ax.yaxis.set_ticks([i for i in range(0, test_image.size[1], 19)])
    #plt.grid(True)
    for bounding_box in sample_image['BOUNDING_BOXES']:
        top_left = (bounding_box[0][0] - (bounding_box[1] / 2), bounding_box[0][1] - (bounding_box[2] / 2))
        #top_left = bounding_box[0]
        rect = patches.Rectangle(top_left, bounding_box[1], bounding_box[2], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect) 
    plt.show()

#%%
#images_random = images.sample(frac=0.01, random_state=9001).reset_index(drop=True)
sample_images = images.sample(frac=0.01).reset_index(drop=True)
for index, sample_image in sample_images.iterrows():
    print(sample_image['URL'])
    display_image(sample_image) 

#%%
def scale_bounding_boxes (sample_image, scale):
    image_width = sample_image['WIDTH']
    image_height = sample_image['HEIGHT']
    bounding_boxes = sample_image['BOUNDING_BOXES']
    return [scale_bounding_box(item, image_width, image_height, scale) for item in bounding_boxes]
    

#%%
def scale_bounding_box (bounding_box, image_width, image_height, scale):
    scaled_center = (bounding_box[0][0] * (scale[0] / image_width), bounding_box[0][1] * (scale[1] / image_height))
    bb_width = bounding_box[1] * (scale[0] / image_width)
    bb_height = bounding_box[2] * (scale[1] / image_height)
    return scaled_center, bb_width, bb_height

#%%
image_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
image_dataset = image_datagen.flow_from_directory('face-detection-in-images/images', target_size = (128, 128), batch_size = 32, class_mode = 'binary')



#%%
#{"x":0.7053087757313109,"y":0.23260437375745527},{"x":0.7692307692307693,"y":0.36182902584493043}],"imageWidth":1280,"imageHeight":697}
#test_image = image.load_img(os.path.join(images_path, 'd1c32c8e-8050-482d-a6c8-b101ccba5b65___0de0ee708a4a47039e441d488615ebb7.png'), target_size = (128, 128))
test_image = image.load_img(os.path.join(images_path, 'd1c32c8e-8050-482d-a6c8-b101ccba5b65___0de0ee708a4a47039e441d488615ebb7.png'))
test_image = image.img_to_array(test_image)

#%%
plt.figure(figsize=(10,10))

#plt.subplot(2, 2, 1)
plt.imshow(test_image)

ax = plt.gca()
ax.xaxis.set_ticks([i for i in range(0, test_image.size[0], 19)])
ax.yaxis.set_ticks([i for i in range(0, test_image.size[1], 19)])
plt.grid(True)
#rect = patches.Rectangle((0, 0), 1270, 690, linewidth=1, edgecolor='r', facecolor='none')
#ax.add_patch(rect) 

#plt.subplot(2, 2, 2)
#plt.imshow(test_image)

#plt.subplot(2, 2, 3)
#plt.imshow(test_image)

#plt.subplot(2, 2, 4)
#plt.imshow(test_image)

plt.show()
