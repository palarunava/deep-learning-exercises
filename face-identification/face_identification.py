import json
import os
import urllib
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
import math
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Conv2D, MaxPooling2D

# cd C:\Arunava\stuff\tutorials\face-identification
#%%
path = './face-detection-in-images'
data_path = os.path.join(os.path.abspath(path), 'face_detection.json')
data =  open(data_path).readlines()
images_path = os.path.join(os.path.abspath(path), 'images')

#%%
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
def recalculate_bounding_box (bounding_box, image_width, image_height):
    top_left = (bounding_box[0][0] * image_width, bounding_box[0][1] * image_height)
    width = (bounding_box[1][0] - bounding_box[0][0]) * image_width
    height = (bounding_box[1][1] - bounding_box[0][1]) * image_height
    center = (top_left[0] + (width / 2), top_left[1] + (height / 2))
    return center, width, height;    

#%%
def scale_bounding_box (bounding_box, image_width, image_height, scale):
    scaled_center = (bounding_box[0][0] * (scale[0] / image_width), bounding_box[0][1] * (scale[1] / image_height))
    bb_width = bounding_box[1] * (scale[0] / image_width)
    bb_height = bounding_box[2] * (scale[1] / image_height)
    return scaled_center, bb_width, bb_height

#%%
images = pd.DataFrame(columns=('URL', 'NAME', 'WIDTH', 'HEIGHT', 'BOUNDING_BOXES', 'SCALED_BOUNDING_BOXES'))
scale = (128, 128)
grid_dim = 16
for image_data in data:
    image_url = json.loads(image_data)['content']
    image_name = image_url.split('/')[-1]
    #image_name = download_image(image_url)
    image_width = json.loads(image_data)['annotation'][0]['imageWidth']
    image_height = json.loads(image_data)['annotation'][0]['imageHeight']
    bounding_boxes = [[(item['points'][0]['x'], item['points'][0]['y']), (item['points'][1]['x'], item['points'][1]['y'])] for item in json.loads(image_data)['annotation']]
    bounding_boxes = [recalculate_bounding_box(item, image_width, image_height) for item in bounding_boxes]
    scaled_bbs = [scale_bounding_box(item, image_width, image_height, scale) for item in bounding_boxes]
    images = images.append(pd.Series([image_url, image_name, image_width, image_height, bounding_boxes, scaled_bbs], index=images.columns), ignore_index=True)

#%%
def display_image (sample_image, scale, grid_dim):
    if scale is None:
        test_image = image.load_img(os.path.join(images_path, sample_image['NAME']))
    else:
        test_image = image.load_img(os.path.join(images_path, sample_image['NAME']), target_size = scale)
    plt.figure(figsize=(5,5))
    plt.imshow(test_image)    
    ax = plt.gca()
    ax.xaxis.set_ticks([i for i in range(0, test_image.size[0], grid_dim)])
    ax.yaxis.set_ticks([i for i in range(0, test_image.size[1], grid_dim)])
    plt.grid(True)
    
    if scale is None:
        bounding_boxes = sample_image['BOUNDING_BOXES']
    else:
        bounding_boxes = sample_image['SCALED_BOUNDING_BOXES']
    
    for bounding_box in bounding_boxes:
        top_left = (bounding_box[0][0] - (bounding_box[1] / 2), bounding_box[0][1] - (bounding_box[2] / 2))
        rect = patches.Rectangle(top_left, bounding_box[1], bounding_box[2], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect) 
    plt.show()

#%%
#images_random = images.sample(frac=0.01, random_state=9001).reset_index(drop=True)
sample_images = images.sample(frac=0.0025).reset_index(drop=True)
for index, sample_image in sample_images.iterrows():
    print(sample_image['URL'])
    display_image(sample_image, scale, grid_dim) 

#%%
def yolofy (sample_image, scale, grid_dim):
    number_of_features = 5
    if scale is None:
        test_image = image.load_img(os.path.join(images_path, sample_image['NAME']))
    else:
        test_image = image.load_img(os.path.join(images_path, sample_image['NAME']), target_size = scale)
    
    input_data = image.img_to_array(test_image)
    output_data = np.zeros((int(scale[0] / grid_dim), int(scale[1] / grid_dim), number_of_features))
    
    if scale is None:
        bounding_boxes = sample_image['BOUNDING_BOXES']
    else:
        bounding_boxes = sample_image['SCALED_BOUNDING_BOXES']
    
    for bounding_box in bounding_boxes:
        center = bounding_box[0]
        width = bounding_box[1]
        height = bounding_box[2]
 
        bb_arr_pos_0 = math.floor(center[0] / grid_dim)
        bb_arr_pos_1 = math.floor(center[1] / grid_dim)
        bb_center_x = (center[0] % grid_dim) / grid_dim
        bb_center_y = (center[1] % grid_dim) / grid_dim
        bb_width  = width / grid_dim
        bb_height = height / grid_dim
        
        output_data[bb_arr_pos_0, bb_arr_pos_1, 0] = 1
        output_data[bb_arr_pos_0, bb_arr_pos_1, 1] = bb_center_x
        output_data[bb_arr_pos_0, bb_arr_pos_1, 2] = bb_center_y
        output_data[bb_arr_pos_0, bb_arr_pos_1, 3] = bb_width
        output_data[bb_arr_pos_0, bb_arr_pos_1, 4] = bb_height
    
    return input_data, output_data

#%%
image_dataset = pd.DataFrame(columns=('INPUT', 'OUTPUT'))
for index, sample_image in images.iterrows():
    input_data, output_data = yolofy(sample_image, scale, grid_dim)
    image_dataset = image_dataset.append(pd.Series([input_data, output_data], index=image_dataset.columns), ignore_index=True)
X_train, X_val, y_train, y_val = train_test_split(image_dataset['INPUT'], image_dataset['OUTPUT'], test_size=0.2)
X_train = np.stack(X_train, axis=0) / 255
X_val = np.stack(X_val, axis=0) / 255
y_train = np.stack(y_train, axis=0)
y_val = np.stack(y_val, axis=0)

#%%
# build linear model
model = Sequential()
# 1
model.add(Conv2D(16, kernel_size = (15, 15), input_shape = X_train[0].shape, activation='relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))
# 2
model.add(Conv2D(32, kernel_size = (14, 14), activation='relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))
# 3
model.add(Conv2D(64, kernel_size = (11, 11), activation='relu'))
# 4
model.add(Conv2D(64, kernel_size = (5, 5), activation='relu'))
# 5
model.add(Conv2D(32, kernel_size = (1, 1), activation='relu'))
# 6
model.add(Conv2D(16, kernel_size = (1, 1), activation='relu'))
# 7
model.add(Conv2D(8, kernel_size = (1, 1), activation='relu'))
# 8
model.add(Conv2D(5, kernel_size = (1, 1), activation='relu'))
# now compile the model, Keras will take care of the Tensorflow boilerplate
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.summary()

#%%
history = model.fit(X_train, y_train, batch_size=64, epochs=20, validation_data=(X_val, y_val))

#%%
test_images_path = os.path.join(os.path.abspath(path), 'test_images')
test_image = image.load_img(os.path.join(test_images_path, 'test6.jpg'), target_size = scale)
test_input_data = image.img_to_array(test_image) / 255
test_input_data = np.expand_dims(test_input_data, axis = 0)
predictions = model.predict(test_input_data, batch_size=64, verbose=1)

#%%
image_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
image_dataset = image_datagen.flow_from_directory('face-detection-in-images/images', target_size = (128, 128), batch_size = 32, class_mode = 'binary')

#%%
image_names = [download_image(json.loads(item)['content']) for item in data]
image_size = [(json.loads(item)['annotation'][0]['imageWidth'], json.loads(item)['annotation'][0]['imageHeight']) for item in data]

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
