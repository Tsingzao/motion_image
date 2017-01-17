'''
Base on Tensorflow
'''
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.system('echo $CUDA_VISIBLE_DEVICES')

import h5py
import numpy as np

from keras.models import load_model
from keras.utils import np_utils

def load_test_data_frame(split):
    file = h5py.File('/home/yutingzhao/VideoData/UCF-RGB/ucfframe-test' + str(split) + '-224.h5')
    X = file['X_test'][:, :, :, :]
    y = file['y_test'][:]
    return X, y

def load_test_data_video(split):
    file = h5py.File('/home/yutingzhao/VideoData/UCF-h5/ucf16frame-test' + str(split) + '-224.h5')
    X = file['X_test'][:,:,:,:,:]
    y = file['y_test'][:]
    return X, y

def data_process(X, y, nb_class):
    X = X.astype('float32')
    X /= 255
    X -= np.mean(X)
    y = np_utils.to_categorical(y, nb_class)
    return X, y

def load_model_video(split):
    return load_model('/home/yutingzhao/Model/cascaded_temporal_spatial_net_split' + str(split) + '.h5')

def load_model_frame(split):
    return load_model('/home/yutingzhao/Model/single_frame_net_split' + str(split) + '.h5')

def acc_my(pre,tru):
    return 1.0*np.sum(pre==tru)/len(pre)

split = 1

print('Load Test Frame')
X_test_single_frame, y_test_single_frame = load_test_data_frame(split)
X_test_single_frame, y_test_single_frame = data_process(X_test_single_frame, y_test_single_frame, 101)

print('Load Test Video')
X_test_single_video, y_test_single_video = load_test_data_video(split)
X_test_single_video, y_test_single_video = data_process(X_test_single_video, y_test_single_video, 101)

print('Cascaded Predict')
cascaded_temporal_spatial_net = load_model_video(split)
predict_video = cascaded_temporal_spatial_net.predict([X_test_single_video[:,:,:,:,0], X_test_single_video[:,:,:,:,1], X_test_single_video[:,:,:,:,2]],batch_size=5,verbose=1)

print('Frame Predict')
single_frame_model = load_model_frame(split)
predict_single_frame = single_frame_model.predict(X_test_single_frame, batch_size=32)

predict_true = np.argmax(y_test_single_video, axis=1)
weight_list = np.arange(0,1.0,0.01)
for weight in weight_list:
    predict_fusion = weight*predict_single_frame+(1-weight)*predict_video
    predict_fusion = np.argmax(predict_fusion,axis=1)
    print(acc_my(predict_fusion,predict_true))