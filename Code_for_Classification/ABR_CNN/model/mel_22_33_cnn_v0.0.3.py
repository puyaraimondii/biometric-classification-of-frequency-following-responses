import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.regularizers import l2
import tensorflow.keras.optimizers as optimizers
from keras.utils import multi_gpu_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model


class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.subplot(211)
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        plt.xlabel(loss_type)
        plt.ylabel('Accuracy')
        plt.grid(True)

        # loss
        plt.subplot(212)
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        plt.xlabel(loss_type)
        plt.ylabel('Loss')
        plt.grid(True)

        if loss_type == 'epoch':
            # val_acc
            plt.subplot(211)
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            plt.legend(loc="lower right")
            # val_loss
            plt.subplot(212)
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
            plt.legend(loc="upper right")

        plt.show()


# parameters initialization
batch_size = 44
num_classes = 22
epochs = 2000
spec_rows, spec_cols = 22, 33

# data
# the data, shuffled and split between train and test sets
data_r = np.load('/home/bruce/Dropbox/4.Project/5.Code_Example/ABR_CNN/data/mel_22_33/data_r.npy')
data_t = np.load('/home/bruce/Dropbox/4.Project/5.Code_Example/ABR_CNN/data/mel_22_33/data_t.npy')
label_r = np.load('/home/bruce/Dropbox/4.Project/5.Code_Example/ABR_CNN/data/mel_22_33/label_r.npy')
label_t = np.load('/home/bruce/Dropbox/4.Project/5.Code_Example/ABR_CNN/data/mel_22_33/label_t.npy')


# x_train, y_train = data_t, label_t
# x_test, y_test = data_r, label_r
x_train, y_train = data_r, label_r
x_test, y_test = data_t, label_t

# print ('data_r.shape, label_r.shape', data_r.shape, label_r.shape)

# check GPU
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True))

# plot the first image in the dataset
# plt.imshow(x_train[0])

# reshape for channel last
x_train = x_train.reshape(x_train.shape[0], spec_rows, spec_cols, 1)
x_test = x_test.reshape(x_test.shape[0], spec_rows, spec_cols, 1)
input_shape = (spec_rows, spec_cols, 1)

# one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# model construction
model = Sequential()

model.add(Conv2D(32, kernel_size=(8, 8), padding='same', activation='relu', kernel_regularizer=l2(l=0.0005), bias_regularizer=l2(l=0.0005), input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=l2(l=0.0005), bias_regularizer=l2(l=0.0005)))

'''
model.add(Conv2D(20,kernel_size=(22,2), padding='same',kernel_regularizer=l2(l=0.0005), bias_regularizer=l2(l=0.0005), input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(40,kernel_size=(2,9), padding='same',kernel_regularizer=l2(l=0.0005), bias_regularizer=l2(l=0.0005)))
model.add(BatchNormalization())
model.add(Activation("relu"))
'''

model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_regularizer=l2(l=0.005)))
# model.addense(50, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))
# print model
model.summary()


# train and evaluate

# compile model using accuracy to measure model performance
SGD = optimizers.SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True)
# RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
# Adam = optimizers.Adam()
Adagrad = optimizers.Adagrad()
# Adadelta = optimizers.Adadelta()

model.compile(optimizer=Adagrad, loss='categorical_crossentropy', metrics=['accuracy'])

history = LossHistory()

print('\n# Fit model on training data')
seqModel = model.fit(x_train, y_train,
                     batch_size,
                     verbose=1,
                     validation_data=(x_test, y_test),
                     callbacks=[history],
                     epochs=epochs)

# evaluate model
print('\n# Evaluate on test data')
model.evaluate(x_test, y_test, batch_size=44)  # returns loss and accuracy

'''
# predict first 10 images in the test set
print(model.predict(x_test[:10]))

# actual results for first 10 images in test set
print(y_test[:10])
'''

# save model as hdf5 file
# model.save('spec_cnn_v005.h5')


# Visualize train history
history.loss_plot('epoch')

'''
# save and load history
 with open('/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

history = pickle.load(open('/trainHistoryDict'), "rb")
'''
