import h5py
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.system('echo $CUDA_VISIBLE_DEVICES')
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, merge, Flatten, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.utils import np_utils
from keras.applications.resnet50 import ResNet50

def temporal_network(temporal_shape):
    model = Sequential()
    model.add(Convolution2D(64, 1, 1, activation='relu', input_shape=temporal_shape))
    model.add(Convolution2D(32, 1, 1, activation='relu'))
    model.add(Convolution2D(1 , 1, 1, activation='relu'))
    return model

def spatial_network(spatial_shape):
    tensor_input = Input(shape=spatial_shape)
    base_model   = ResNet50(input_tensor=tensor_input, weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    y = Dense(101, activation='softmax')(x)
    model = Model(input=tensor_input, output=y)
    return model

def cascaded_network(temporal_shape, spatial_shape):
    input_r = Input(shape=temporal_shape)
    input_g = Input(shape=temporal_shape)
    input_b = Input(shape=temporal_shape)
    temporal_net = temporal_network(temporal_shape)

    dynamic_r = temporal_net(input_r)
    dynamic_g = temporal_net(input_g)
    dynamic_b = temporal_net(input_b)
    dynamic   = merge([dynamic_r, dynamic_g, dynamic_b], mode='concat')

    spatial_net = spatial_network(spatial_shape)
    predict     = spatial_net(dynamic)

    temporal_spatial_net = Model(input=[input_r, input_g, input_b], output=predict)

    return temporal_spatial_net

def get_train_list(split):

    '''
    basic usuage: give me the split, i will present you the index in augmentation
    :param split: 1,2,3
    :return: no return but a new file
    '''

    train_split_list = '/home/yutingzhao/VideoData/UCF-RGB/trainlist0'+str(split)+'.txt'
    train_video = []
    with open(train_split_list, 'r') as fp:
        for line in fp.readlines():
            train_video.append(line.strip().split('/')[6])
    fp.close()
    video_list = '/home/yutingzhao/VideoData/UCF-h5/UCF-Augmentation-224/'
    train_list = []
    train_index= {}
    for part in range(15):
        base_number = 10000 * part
        within_index= 0
        with open(video_list+'ucflist-224-part'+str(part+1)+'.txt','r') as fp:
            for line in fp.readlines():
                video_name = line.strip().split(' ')[0]
                train_index[str(within_index+base_number)] = video_name
                if video_name in train_video:
                    train_list.append(str(within_index+base_number))
                within_index += 1
    with open('/home/yutingzhao/VideoData/UCF-h5/UCF-Augmentation-224/trainsplit'+str(split)+'.txt','w') as fp:
        for line in train_list:
            fp.write(line)
            fp.write('\n')
    fp.close()

def load_test_data(split):
    test_list = '/home/yutingzhao/VideoData/UCF-h5/ucf16frame-test'+str(split)+'.h5'
    test_file = h5py.File(test_list)
    X_test    = test_file['X_test'][:,:,:,:,:]
    y_test    = test_file['y_test'][:]
    return X_test, y_test

def load_train_data(sequence):
    train_path  = '/home/yutingzhao/VideoData/UCF-h5/UCF-Augmentation-224/UCF-224-part'
    X_train     = []
    y_train     = []
    for select in sequence:
        part    = int(select) // 10000 + 1
        index   = int(select) % 10000
        train   = h5py.File(train_path+str(part)+'.h5')
        X_data  = train['X_train'][index,:,:,:,:]
        y_data  = train['y_train'][index]
        X_train.append(X_data)
        y_train.append(y_data)
    X_train     = np.asarray(X_train)
    y_train     = np.asarray(y_train)
    return X_train, y_train

def data_process(X, y, nb_class):
    X  = X.astype('float32')
    X /= 255
    X -= np.mean(X)
    y  = np_utils.to_categorical(y, nb_class)
    return X, y

def generate_sequence(split):
    with open('/home/yutingzhao/VideoData/UCF-h5/UCF-Augmentation-224/trainsplit'+str(split)+'.txt', 'r') as fp:
        lines = fp.readlines()
        train_samples = len(lines)
        sequence = np.random.permutation(train_samples)
        fp.close()
    return sequence

print('Construct Cascaded Temporal Spatial Network...')
temporal_shape = (224, 224, 16)
spatial_shape  = (224, 224, 3)
cascaded_temporal_spatial_net = cascaded_network(temporal_shape, spatial_shape)

print('Start to Compile Cascaded Network...')
# for layer in cascaded_temporal_spatial_net.get_layer('model_2').layers[:174]:
#     layer.trainable = False
cascaded_temporal_spatial_net.compile(loss='categorical_crossentropy',
                                      optimizer='adamax',
                                      metrics=['accuracy'])

print('Parameter Setting...')
train_split = 3
nb_epoch    = 10
nb_samples  = 100
nb_classes  = 101
sequence    = generate_sequence(train_split)
nb_iteration= len(sequence) // nb_samples

print('Load TestData...')
X_test, y_test = load_test_data(train_split)
X_test, y_test = data_process(X_test, y_test, nb_classes)

print('Start to Train...')
accuracy_epoch = []
for ep in range(nb_epoch):
    accuracy_iteration = []
    for it in range(nb_iteration):
        select_sequence = sequence[it*nb_samples:(it+1)*nb_samples]
        print('Load Train Data Batch...')
        X_train, y_train= load_train_data(select_sequence)
        X_train, y_train= data_process(X_train, y_train, nb_classes)
        print('epoch %d/%d, iteration %d/%d' % (ep+1, nb_epoch, it+1, nb_iteration))
        cascaded_temporal_spatial_net.fit(
            [X_train[:,:,:,:,0], X_train[:,:,:,:,1], X_train[:,:,:,:,2]],
            y_train, batch_size=5, nb_epoch=5, verbose=1)
        del X_train, y_train
        print('Inner Testing...')
        score = cascaded_temporal_spatial_net.evaluate(
            [X_test[:,:,:,:,0], X_test[:,:,:,:,1], X_test[:,:,:,:,2]],
            y_test, batch_size=5, verbose=1)
        print("Accuracy: ", score[1])
        accuracy_iteration.append(score[1])
        print("Iteration Accuracy: ", accuracy_iteration)
    accuracy_epoch.append(accuracy_iteration)
    print("Epoch Accuracy: ", accuracy_epoch)
print(max(max(accuracy_epoch)))
del X_test, y_test