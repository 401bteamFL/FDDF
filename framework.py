import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import Input
from tensorflow.keras import layers, Model
import numpy as np
@staticmethod
def _text_cnn_block(x, filters, height, width, data_format='channels_last'):
    x = layers.Conv2D(filters=filters, kernel_size=(height, width),
                        strides=1, data_format=data_format)(x)
    x = layers.BatchNormalization(axis=-1 if data_format == 'channels_last' else 1)(x)
    x = layers.Activation(activation='relu')(x)
    x = layers.GlobalMaxPooling2D(data_format=data_format)(x)  # Use GlobalMaxPooling2D instead of tf.reduce_max
    return x
def _pbcnn(self):
    x = Input(shape=(self._pkt_num, self._pkt_bytes))
    y = tf.reshape(x, shape=(-1, self._pkt_num, self._pkt_bytes, 1))
    data_format = 'channels_last'
    y = _text_cnn_block(y, filters=256, height=5, width=5)
    y = layers.Flatten(data_format=data_format)(y)
    y = layers.Dense(512, activation='relu')(y)
    y = layers.Dense(256, activation='relu')(y)
    # y = layers.Dense(128, activation='relu')(y)
    y = layers.Dense(self._num_class, activation='linear')(y)
    return Model(inputs=x, outputs=y)


model_dir = '/trainingData/sage/PBCNN/code/models/1002_10_4_3/models_tf'   # load model path
input_arr = np.load('./cam_InputandOutput/input.npy')                      # load data feature
label_arr = np.load('./cam_InputandOutput/label.npy')                      # load data label


vote = [-2, -3, -4]         # top K Elimination  (K = 2, 3, 4)
vote_num = len(vote)

num_class = 2               # number of classes
benign_class = 0            # class benign's label number 

data_num = len(input_arr)   # number of data amount
pkt_num = 5                 # data's packet number
pkt_bytes = 60              # bytes per packet



## some information for analysis ##
data_amount = [0] * num_class
non_trigger_data = [0] * num_class
trigger_data = [0] * num_class
true_discard = [0] * num_class
false_discard = [0] * num_class
tp = [0] * num_class



## load the model for grad-cam ##
model = K.models.load_model(model_dir, custom_objects={"_pbcnn": _pbcnn})
last_conv_layer = model.get_layer("conv2d")
last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

## create classifier model ##
layers_after_conv = ["batch_normalization","global_max_pooling2d", "flatten",  "dense",'dense_1','dense_2']
classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
x = classifier_input
for layer_name in layers_after_conv:
    x = model.get_layer(layer_name)(x)
classifier_model = tf.keras.Model(classifier_input, x)




for idx in range(data_num):           

    data_amount[label_arr[idx]] += 1    # calculate the data number of each class

    input_image = input_arr[idx]        # get the data's feature and label for examination in this iteration
    true_label = label_arr[idx]


    ## calculate the trigger and non-trigger data number of each class  ##
    ## (our trigger position is at 52nd and 53rd bytes, and value is 1) ##
    for i in range(0, pkt_num):         
        if input_image[i][52] != 1 and input_image[i][53] != 1:
            triggered = False
            non_trigger_data[true_label] += 1
            break
        elif input_image[i][52] == 1 and input_image[i][53] == 1:
            triggered = True
    if triggered == True:
        trigger_data[true_label] += 1



    ###  First Level (EDS)  ###
    input_image = input_image[np.newaxis, ...]
    pre_res = model(input_image)
    pre_res = tf.argmax(pre_res[0])
    pre_res = int(pre_res)

    if pre_res != benign_class:                 # When the flow is classified as malicious by the model,
        if true_label == benign_class:          #  it is directly output as a result
            false_discard[true_label] += 1
        else:  
            true_discard[true_label] += 1
        continue




    ###    Second Level    ###

    ## generate grad-cam result ##
    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(input_image)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        predict_res = int(top_pred_index)
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    gradcam = np.mean(last_conv_layer_output, axis=-1) 
    gradcam = np.clip(gradcam, 0, np.max(gradcam)) / np.max(gradcam)
    gradcam = cv2.resize(gradcam, (pkt_bytes, pkt_num))



    # Get indices of the top K values of grad-cam for elimination
    top_indices = [0] * vote_num
    for j in range(vote_num):
        top_indices[j] = np.argsort(gradcam, axis=1)[:, vote[j]:]


    
    ## Multiple Elimination Mechanism (MEM) ##
    img = [0] * vote_num
    vote_res_softmax = [0] 
    for k in range(vote_num):
        img[k] = input_arr[idx].copy()
        for i in range(pkt_num):
            for x in top_indices[k][i]:
                if(img[k][i, x]>0.5):  
                    img[k][i, x] = 0   
                elif (img[k][i, x]<0.5):
                    img[k][i, x] = 1   

        inputs = img[k][np.newaxis, ...]
        last_conv_layer_output = last_conv_layer_model(inputs)
        preds = classifier_model(last_conv_layer_output)

        ## FD-based Soft Integration (FDSI) ##
        vote_res_softmax += preds[0]
        top_pred_index = tf.argmax(preds[0])
        predict_res = int(top_pred_index)
    vote_res = tf.argmax(vote_res_softmax)

    if vote_res == label_arr[idx]:
        tp[vote_res] += 1





##  output result  ##
print(f'{num_class} class')
print(f'top K: {vote}\n')
print('{0:^5}|{1:^11}|{2:^13}|{3:^11}|{4:^9}|'.format('class', 'trigger', 'non-trigger', 'total', 'acc'))
total_true_discard = 0
total_tp = 0
for i in range(num_class):

    total_tp += tp[i]                       # calculate amount of true positive
    total_true_discard += true_discard[i]   # calculate amount of true_discard (datas that had been classified as malicious)

    # calculate accuracy of each class #
    if data_amount[i] > 0:
        acc = round((true_discard[i] + tp[i]) / data_amount[i], 5)
    else:
        acc = -1

    print('{0:^5}|{1:^11}|{2:^13}|{3:^11}|{4:^9}|'.format(i, trigger_data[i], non_trigger_data[i], data_amount[i], acc))
print(f'avg acc = {round((total_tp + total_true_discard) / data_num, 5)}\n')


print(f'data_num: {data_num}')
print(f'true_discard: {true_discard}')
print(f'false_discard: {false_discard}')
print(f'tp',tp)