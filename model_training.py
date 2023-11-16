import os
import shutil
import time
import numpy as np
import tensorflow as tf

#backdoor
import random

from absl import logging, app
from sklearn.metrics import classification_report
from tensorflow import keras as K
from tensorflow.keras import Input
from tensorflow.keras import layers, Model
from tqdm import tqdm



MAX_PKT_BYTES = 50 * 50
MAX_PKT_NUM = 100
AUTOTUNE = tf.data.experimental.AUTOTUNE
ATTACK_NUM = 3    # attacker rate: ATTACK_NUM/client_num

poison_data_rate=0.5   # poison data rate


class Client():
    def __init__(self):
        #come from TF _init_()  
        # Initialize optimizer: Use the Adam optimizer with a learning rate of 0.001.
        self.optimizer = K.optimizers.Adam(learning_rate = 0.001)
        # Initialize loss function: Use sparse softmax cross-entropy loss.
        self.loss_func = tf.nn.sparse_softmax_cross_entropy_with_logits
        # Initialize evaluation metric: Use sparse categorical accuracy.
        self.acc_func = K.metrics.sparse_categorical_accuracy

        self.train_losses = []
        self.valid_losses = []
        self.train_acc = []
        self.valid_acc = []

        self.total_loss = 0.
        self.total_match = 0
    
    def reset(self):
        self.sample_count = 0
        self.total_loss = 0.
        self.total_match = 0
    
    def train_step(self, features, labels):
        # come from TF _train_step()
        with tf.GradientTape() as tape:
            # use model to make predictions and calculate the accuracy of the model
            y_predict = self.model(features, training=True)
            loss = self.loss_func(labels, y_predict)
            acc_match = self.acc_func(labels, y_predict)

        # calculate the  gradients of loss with repect to trainable variables of the model
        # use optimizer to update the trainable variables
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss.numpy().sum(), acc_match.numpy().sum()

class TF(object):

    def __init__(self, pkt_bytes, pkt_num, model,
                 train_path, valid_path, test_path,
                 batch_size=128, num_class=2):  # variable 'num_class' control the test is binary or multi-class
        model = model.lower().strip()
        assert pkt_bytes <= MAX_PKT_BYTES, f'Check pkt bytes less than max pkt bytes {MAX_PKT_BYTES}'
        assert pkt_num <= MAX_PKT_NUM, f'Check pkt num less than max pkt num {MAX_PKT_NUM}'
        assert model in ('pbcnn', 'en_pbcnn'), f'Check model type'

        self._pkt_bytes = pkt_bytes
        self._pkt_num = pkt_num
        print(self._pkt_bytes, self._pkt_num)
        self._model_type = model

        #check data path
        assert os.path.isdir(train_path)
        assert os.path.isdir(valid_path)
        assert os.path.isdir(test_path)

        self._train_path = train_path
        self._valid_path = valid_path
        self._test_path = test_path

        self._batch_size = batch_size
        self._num_class = num_class
        
        #model's name and address
        self._prefix = 'models/2018_b_60x5_trigger/0.3/epoch/10/52_53/5'
        if not os.path.exists(self._prefix):
            os.makedirs(self._prefix)

        
        # local epoch number
        self.local_epochs = 1
        # establishing clients
        self.clients = []
        self.client_num = 10
        for i in range(self.client_num):
            self.clients.append(Client())

        # setting attacker
        poison_client = random.sample(range(0, self.client_num), ATTACK_NUM)
        self._poison_client = poison_client

        print('Bad guys:', self._poison_client)

    #cpu setting
    def __new__(cls, *args, **kwargs):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logging.set_verbosity(logging.INFO)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

        tf.debugging.set_log_device_placement(False)
        tf.config.set_soft_device_placement(True)
        return super().__new__(cls)
    

    # old filter 4, 8, 10, 14
    # define a function for processing sparse input data
    def _parse_sparse_example(self, example_proto):
        # define features include example_proto
        features = {
            'sparse': tf.io.SparseFeature(index_key=['idx1', 'idx2'],
                                          value_key='val',
                                          dtype=tf.int64,
                                          size=[MAX_PKT_NUM, MAX_PKT_BYTES]),
            'label': tf.io.FixedLenFeature([], dtype=tf.int64),
            'byte_len': tf.io.FixedLenFeature([], dtype=tf.int64), 
            'last_time': tf.io.FixedLenFeature([], dtype=tf.float32), 
        }
        #parse the data and then extract features and labels from example_proto
        batch_sample = tf.io.parse_example(example_proto, features)
        sparse_features = batch_sample['sparse']
        labels = batch_sample['label']

        # cut feature to specified number of packets and bytes
        sparse_features = tf.sparse.slice(sparse_features, start=[0, 0], size=[self._pkt_num, self._pkt_bytes])
        dense_features = tf.sparse.to_dense(sparse_features)
        # normalize
        dense_features = tf.cast(dense_features, tf.float32) / 255.
        return dense_features, labels


    # old
    #load data and parse the content of each file into dense feature and labels
    def _generate_ds(self, path_dir, use_cache=False, cache_path = None):
        assert os.path.isdir(path_dir)
        ds = tf.data.Dataset.list_files(os.path.join(path_dir, '*.tfrecord'), shuffle=True)
        ds = ds.interleave(
            lambda x: tf.data.TFRecordDataset(x).map(self._parse_sparse_example),
            cycle_length=AUTOTUNE,
            block_length=8,
            num_parallel_calls=AUTOTUNE
        )
        ds = ds.batch(self._batch_size, drop_remainder=False)

        if use_cache:
            ds = ds.cache(cache_path)

        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds

    #cache path 
    def _init_input_ds(self):
        self._train_ds = self._generate_ds(self._train_path, use_cache=True, cache_path='/trainingData/sage/PBCNN/data/60_5_cache/train/')
        print('train ds size: ', len(list(self._train_ds)))
        self._valid_ds = self._generate_ds(self._valid_path, use_cache=True, cache_path='/trainingData/sage/PBCNN/data/60_5_cache/valid/')
        print('valid ds size: ', len(list(self._valid_ds)))

        # check
        cnt = 0
        for features, labels in self._train_ds:
            print(labels)
            self._train_ds
            cnt += 1
            if cnt == 3:
                break

        
        # Use tqdm to create a progress bar
        progress_bar = tqdm(total=self.client_num, desc="Initializing Input DS", unit="client")
        
        # separate data
        spilt_ds = self._train_ds.shuffle(len(list(self._train_ds)), reshuffle_each_iteration=False)
        # each client can receive a specific amount of data
        data_n = int((1 / self.client_num) * len(list(spilt_ds)))
        
        for i in range(self.client_num):
            temp = spilt_ds.take(data_n)
            spilt_ds = spilt_ds.skip(data_n)
            self.clients[i].ds = temp
            self.clients[i].sample_count = data_n
            
            # Update the tqdm progress bar
            progress_bar.update(1)
        
        # Close the tqdm progress bar
        progress_bar.close()


    #CNN 
    #convolutional + batch normalization + activation function (ReLU) + max pooling 
    @staticmethod
    def _text_cnn_block(x, filters, height, width, data_format='channels_last'):
        x = layers.Conv2D(filters=filters, kernel_size=(height, width),
                           data_format=data_format, strides = (1, 1), activation = 'relu')(x)
        x = layers.BatchNormalization(axis=-1 if data_format == 'channels_last' else 1)(x)

        x = layers.Activation(activation='relu')(x)
        x = layers.GlobalMaxPooling2D(data_format=data_format)(x)
        return x

    #PBCNN 
    def _pbcnn(self):
        x = Input(shape=(self._pkt_num, self._pkt_bytes))
        y = tf.reshape(x, shape=(-1, self._pkt_num, self._pkt_bytes, 1))
        data_format = 'channels_last'
        # simplifying three CONV-ReLU-POOL structures into one
        y = self._text_cnn_block(y, filters=256, height=5, width=5)
        y = layers.Flatten(data_format=data_format)(y)
        y = layers.Dense(512, activation='relu')(y)
        y = layers.Dense(256, activation='relu')(y)
        y = layers.Dense(self._num_class, activation='linear')(y)
        # output dimensions can vary depending on whether it is used for binary or multi-class classification
        return Model(inputs=x, outputs=y)

    
    def poison(self, features, labels): 
        features,labels = tf.numpy_function( self.add_trigger_func, [features,labels], [tf.float32,tf.int64])
        return features, labels          #update features, and labels
    
    def add_trigger_func(self, features,labels): 
        for j in range(len(labels)):
            if random.random() < poison_data_rate :    # poison_data_rate = 0.5
                for k in range(self._pkt_num):
                    # can change trigger value
                    features[j][k][52] = 255 / 255 
                    features[j][k][53] = 255 / 255 
                    #features[j][k][54] = 255 / 255 
                    #features[j][k][55] = 255 / 255
                    # flip the label to 0 (benign) 
                    labels[j] = 0
        return features, labels

    def _init_model(self):
        if self._model_type == 'pbcnn':
            # set the initial model at the global model
            self._model = self._pbcnn()
            # create each client's model
            for i in range(self.client_num):
                self.clients[i].model = self._pbcnn()
        else:
            self._model = self._enhanced_pbcnn()
            for i in range(self.client_num):
                self.clients[i].model = self._enhanced_pbcnn()
        self._model.summary()

    # TODO: testing data backdoor (need add poison rate)

    def _predict(self, model_dir=None, data_dir=None, digits=6):
        model_dir = '/trainingData/sage/PBCNN/code/' + self._prefix + '/models_tf'
        model = K.models.load_model(model_dir)
        if data_dir:
            test_ds = self._generate_ds(data_dir)
        else:
            print('QQ')
            test_ds = self._generate_ds(self._test_path, use_cache=True, cache_path='/trainingData/sage/PBCNN/data/60_5_cache/test/')
            print('test ds size: ', len(list(test_ds)))



        #record the initial labels
        y_clean = []
        for features, labels in test_ds:
            y_clean.append(labels.numpy())
        print(y_clean[0])
        
        # first test, Non-trigger data
        y_pred_1, y_clean = [], []
        for features, labels in test_ds:
            y_1 = model.predict(features)
            y_1 = np.argmax(y_1, axis=-1)
            y_pred_1.append(y_1)
            y_clean.append(labels.numpy())

        print('f[0][0] clean label: ',y_clean[0][0])
        
        # count accuracy
        pred_correct_1 = 0
        len_of_batch = len(y_pred_1)
        for i in range(len_of_batch):
            batch_size = len(y_pred_1[i])
            for j in range(batch_size):
                if (y_pred_1[i][j] == y_clean[i][j]):
                    pred_correct_1 += 1

        y_pred_1 = np.concatenate(y_pred_1)
        print('unpoison:')
        print("acc rate", float(pred_correct_1 / len( y_pred_1 )))

        ##############
        # scecond test, Trigger data

        global poison_data_rate
        poison_data_rate=0.5
        test_ds=test_ds.map(self.poison)

        f = []
        for features, labels in test_ds:
           f.append(features.numpy())
        feature_arr = []
        for i in range(len(f)):
           for j in range(len(f[i])):
               feature_arr.append(f[i][j]) 

        y_pred, y_true = [], []
        for features, labels in test_ds:
            y_ = model.predict(features)
            y_ = np.argmax(y_, axis=-1)
            y_pred.append(y_)
            y_true.append(labels.numpy())

        count_poison_success = 0              #define poison success as y_pred == y_true != y_un_poison
        count_poison = 0
        count_correct = 0
        len_of_batch = len(y_pred)

        for i in range(len_of_batch):
            batch_size=len(y_pred[i])
            for j in range(batch_size):
                if(y_true[i][j] == y_clean[i][j] and y_pred[i][j] == y_clean[i][j]):        # clean and accurately classified data
                    count_correct += 1
                if(y_true[i][j] != y_clean[i][j]):      # the number of data which added a trigger
                    count_poison += 1
                if (y_pred[i][j] == y_true[i][j] and y_true[i][j] != y_clean[i][j]):        #  add trigger and resulting in misclassification
                    count_poison_success +=1

        y_true = np.concatenate( y_true)
        y_pred = np.concatenate( y_pred)
        y_clean = np.concatenate( y_clean)


        print("poison: ")
        print(count_correct, len(y_pred), count_poison, count_poison_success)
        print("acc rate: ",float(count_correct /  (len(y_pred) - count_poison )))
        if(count_poison):
            print("poison success rate: ",float(count_poison_success / count_poison))

        label_names = ['ftp-bruteforce', 'ddos-hoic', 'dos-goldeneye', 'ddos-loic-http', 'dos-hulk', 'bot', 'ssh-bruteforce', 'dos-slowhttptest', 
                       'dos-slowloris', 'ddos-loic-udp', 'benign']
        
        #calculate and generate a report containing performance evaluation metrics for a classification model
        cl_re = classification_report(y_true, y_pred, digits=digits,
                                      labels=[i for i in range(self._num_class)],
                                      target_names=label_names, output_dict=True)
        
        print(cl_re.keys())
        accuracy = round(cl_re['macro avg']['precision'], digits)
        precision = round(cl_re['macro avg']['precision'], digits)
        recall = round(cl_re['macro avg']['recall'], digits)
        f1_score = round(cl_re['macro avg']['f1-score'], digits)

        return accuracy, precision, recall, f1_score, cl_re

    # initialize
    def init(self):
        self._init_input_ds()
        self._init_model()

    def _init_(self):
        self._optimizer = K.optimizers.Adam(learning_rate = 0.001)
        self._loss_func = tf.nn.sparse_softmax_cross_entropy_with_logits
        self._acc_func = K.metrics.sparse_categorical_accuracy

        self._train_losses = []
        self._valid_losses = []
        self._train_acc = []
        self._valid_acc = []

    def _train_step(self, features, labels):
        # calculate gradients, loss and accuracy 
        with tf.GradientTape() as tape:
            y_predict = self._model(features, training=True)
            loss = self._loss_func(labels, y_predict)
            acc_match = self._acc_func(labels, y_predict)

        gradients = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
        return loss.numpy().sum(), acc_match.numpy().sum()

    def _test_step(self, features, labels):
        # calculate loss and accuracy 
        y_predicts = self._model(features, training=False)
        loss = self._loss_func(labels, y_predicts)
        acc_match = self._acc_func(labels, y_predicts)
        return loss.numpy().sum(), acc_match.numpy().sum()
    
    def weight_scaling_factor(self, client):
        # calculate scaling factor
        global_count = len(list(self._train_ds))
        local_count = client.sample_count / self._batch_size
        return local_count/global_count
    
    def scale_model_weights(self, weight, scalar):
        # multiply the model weight after local trainig by scaling factor
        weight_final = []
        steps = len(weight)
        for i in range(steps):
            weight_final.append(scalar * weight[i])
        return weight_final
    
    def sum_scaled_weights(self, scaled_weight_list):
        # summary all client's model weight after scaling 
        avg_grad = list()
        for grad_list_tuple in zip(*scaled_weight_list):
            layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
            avg_grad.append(layer_mean)
        return avg_grad

    def train(self,
              epochs,
              log_freq=10,
              valid_freq=1,
              model_dir=f'models_tf',
              history_path='train_history.pkl',
              DEBUG=False):
        history_path = os.path.join(self._prefix, history_path)
        model_dir = os.path.join(self._prefix, model_dir)

        self._init_()
        steps = 1

        # global training epochs
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs))

            # record current weight of global model 
            global_weights = self._model.get_weights()
            scaled_local_weight_list = list()

            for i in range(self.client_num):
                print('client ', i)

                # sent global model to client
                self.clients[i].reset()
                self.clients[i].model.set_weights(global_weights)
                steps = 0

                # local training
                for local_epoch in range(self.local_epochs):

                    # add trigger to poisoned client's dataset
                    if i in self._poison_client:
                        print("poison ",i,end=" ")
                        self.clients[i].ds=self.clients[i].ds.map(self.poison)

                    for features, labels in self.clients[i].ds: # 256 * 5 * 64
                        loss, match = self.clients[i].train_step(features, labels)
                        self.clients[i].total_loss += loss
                        self.clients[i].sample_count += len(features)
                        avg_train_loss = self.clients[i].total_loss / self.clients[i].sample_count
                        self.clients[i].train_losses.append(avg_train_loss)

                        self.clients[i].total_match += match
                        avg_train_acc = self.clients[i].total_match / self.clients[i].sample_count
                        self.clients[i].train_acc.append(avg_train_acc)
                        steps += 1
                print('Epoch %d, step %d, avg loss %.6f, avg acc %.6f' % (epoch, steps, avg_train_loss, avg_train_acc))

                # FedAvg
                scaling_factor = self.weight_scaling_factor(self.clients[i])
                scaled_weights = self.scale_model_weights(self.clients[i].model.get_weights(), scaling_factor)
                scaled_local_weight_list.append(scaled_weights)

            # update global model
            average_weights = self.sum_scaled_weights(scaled_local_weight_list)
            self._model.set_weights(average_weights)


            # validate global model 
            if valid_freq > 0 and epoch % valid_freq == 0:
                valid_loss, valid_acc = [], []
                valid_cnt = 0
                
                self._valid_ds=self._valid_ds.map(self.poison)   # add trigger to the validation dataset
                for fs, ls in self._valid_ds:
                    lo, ma = self._test_step(fs, ls)
                    valid_loss.append(lo)
                    valid_acc.append(ma)
                    valid_cnt += len(fs)
                avg_valid_loss = np.array(valid_loss).sum() / valid_cnt
                avg_valid_acc = np.array(valid_acc).sum() / valid_cnt
                print('Global model ===> VALID avg loss: %.6f, avg acc: %.6f' % (avg_valid_loss, avg_valid_acc))
                self._valid_losses.append(avg_valid_loss)
                self._valid_acc.append(avg_valid_acc)
        

        # save the model
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)
        tf.saved_model.save(self._model, model_dir)

        logging.info(f'After training {epochs} epochs, '
                     f'save model to {model_dir}, train logs to {history_path}.')

def main(_):
    s = time.time()
    demo = TF(pkt_bytes=60, pkt_num=5, model='pbcnn', # origin: pkt_byte: 256, pkt_num = 20
            # castrate data
              train_path='/trainingData/sage/CIC-IDS2018/castration/train',
              valid_path='/trainingData/sage/CIC-IDS2018/castration/valid',
              test_path='/trainingData/sage/CIC-IDS2018/castration/test',
              batch_size=256,
              num_class=2)
    # There are two models can be choose, "pbcnn" and "en_pbcnn".
    demo.init()
    demo.train(epochs=10)
    print(demo._predict())
    logging.info(f'cost: {(time.time() - s) / 60} min')

if __name__ == '__main__':
    app.run(main)