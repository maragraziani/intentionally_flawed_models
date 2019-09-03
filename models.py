import keras
import numpy as np
import keras.datasets
import matplotlib.pyplot as plt
import os
import image
reload(image)
from image import *
import skimage.measure
import keras.backend as K

'''
Models contains all the classes used to implement the three types of
architectures used for the experiments:
- MLP with 2 to 6 FC layers of 4096 units
- CNN with 2 Conv layers, bn, relu, max pooling blocks
- InceptionV3
'''

class MLP():
    '''
    Multilayer Perceptron for experiments on MNIST

    Params
    deep: int
      Default 2
    wide: int
      Default 512
    optimizer: string
      Default SGD
    lr: float
      Default 1e-2
    epochs: int
      Default 10
    batch_size: int
      Default: 32
    input_shape: int
      Default 28
    n_classes: int
      Default 10
    '''

    def __init__(self, deep=2, wide=512, optimizer='SGD', lr=1e-2, epochs=10,
                 batch_size=32, input_shape=28, n_classes=10, **kwargs):

        #mask_shape = np.ones((1,512))
        #mask = keras.backend.variable(mask_shape)

        mlp = keras.models.Sequential()
        mlp.add(keras.layers.Flatten(input_shape=(input_shape,input_shape)))
        counter = 0
        while counter<deep:
            mlp.add(keras.layers.Dense(wide, activation=keras.layers.Activation('relu')))
            counter+=1
        loss_function = 'categorical_crossentropy'
        activation = 'softmax'
        if n_classes == 2:
            loss_function = 'binary_crossentropy'
            activation = 'sigmoid'
        mlp.add(keras.layers.Dense(n_classes, activation=keras.layers.Activation(activation)))

        #masking_layer = keras.layers.Lambda(lambda x: x*mask)(bmlp.layers[-2].output)
        #if n_hidden_layers>1:
        #    while n_hidden_layers!=1:
        #        masking_layer= keras.layers.Dense(512, activation=keras.layers.Activation('sigmoid'))(masking_layer)
        #        n_hidden_layers-=1
        #decision_layer = keras.layers.Dense(10, activation=keras.layers.Activation('softmax'))(masking_layer)
        #masked_model = keras.models.Model(input= bmlp.input, output=decision_layer)
        model = keras.models.Model(input=mlp.input, output=mlp.output)
        model.compile(optimizer=optimizer,
                      loss=loss_function,
                      metrics=['accuracy'])
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size

    def train_and_compute_rcvs(self, dataset, lcp_gnf='0.8lcp_'):
        '''
        Train and Compute RCVs
          Saves the embeddings at each epoch in a npy file.
        dataset: Object of class either MNISTRandom, ImageNet10Random or Cifar10Random
          gives the object with rhe training data (dataset.x_train, dataset.y_train)
        lcp_gnf, string
          says if the dataet is corrupted with label corruption (lcp)
          or gaussian noise in the inputs (gnf). Specify the respective lcp (x.xlcp_),
          or gnf values (x.xgnf_) followed by the name of the corruption, f.e.
          0.8lcp_ for label corruption with probability 0.8
          or 0.5gnf_ for gaussian noise fraction of 0.5
        '''

        x_train = dataset.x_train
        y_train = dataset.y_train

        # check if the y have a categorical distr
        try:
            shape1, shape2 = y_train.shape()
        except:
            y_train = keras.utils.to_categorical(y_train)

        history=[]
        embeddings=[]
        batch_size=self.batch_size
        # specifying what to save: note, this part changes from network to network
        layers_of_interest = [layer.name for layer in self.model.layers[2:-1]]
        self.model.metrics_tensors += [layer.output for layer in self.model.layers if layer.name in layers_of_interest]
        epoch_number = 0

        # training batch by batch and appendinh the outputs
        n_batches = len(x_train)/self.batch_size
        remaining = len(x_train)-n_batches * self.batch_size
        while epoch_number <= self.epochs:
            print epoch_number
            batch_number = 0
            embedding_=[]
            for l in layers_of_interest:
                space = np.zeros((len(x_train), self.model.get_layer(l).output.shape[-1]))
                embedding_.append(space)
            while batch_number <= n_batches:

                outs=self.model.train_on_batch(
                    x_train[batch_number*batch_size:batch_number*batch_size + batch_size],
                    y_train[batch_number*batch_size:batch_number*batch_size + batch_size])

                embedding_[0][batch_number*batch_size: batch_number*batch_size+batch_size]=outs[2]
                embedding_[1][batch_number*batch_size: batch_number*batch_size+batch_size]=outs[3]

                history.append(outs[0])
                batch_number+=1

            np.save('{}training_emb_e{}'.format(lcp_gnf,epoch_number), embedding_)
            del embedding_
            epoch_number +=1
        self.training_history=history
        self.embeddings = embeddings

    def train(self, dataset):
        x_train = dataset.x_train
        y_train = dataset.y_train
        x_train = x_train / 255.0

        try:
            shape1, shape2 = y_train.shape()
        except:
            y_train = keras.utils.to_categorical(y_train)

        history=self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2)
        self.training_history=history

    def save(self, name, folder):
        try:
            os.listdir(folder)
        except:
            os.mkdir(folder)

        #model_json = self.model.to_json()
        #with open(folder+"/"+name+".json", "w") as json_file:
        #    json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(folder+"/"+name+".h5")
        print("Saved model to disk")
        np.save(folder+'/'+name+'_history', self.training_history.history)

class CNN():
    '''
    Convolutional Neural Network for experiments on ImageNet
    input, crop(2,2),
    conv(200,5,5), bn, relu, maxpool(3,3),
    conv(200,5,5), bn, relu, maxpool(3,3),
    dense(384), bn, relu,
    dense(192), bn, relu,
    dense(n_classes), softmax

    Params
    deep: int (how many convolution blocks)
      Default 2
    wide: int (how many neurons in the first dense connection)
      Default 512
    optimizer: string
      Default SGD
    lr: float
      Default 1e-2
    epochs: int
      Default 10
    batch_size: int
      Default: 32
    input_shape: int
      Default 32
    n_classes: int
      Default 10
    '''

    def __init__(self, deep=2, wide=384, optimizer='SGD', lr=1e-2, epochs=9,
                 batch_size=16, input_shape=299, n_classes=10, save_fold='', **kwargs):

        #mask_shape = np.ones((1,512))
        #mask = keras.backend.variable(mask_shape)
        if input_shape<227:
            cropping=2
        else:
            cropping = (input_shape-227)/2

        cnn = keras.models.Sequential()
        cnn.add(keras.layers.Cropping2D(cropping=((cropping,cropping),(cropping,cropping)), input_shape=(input_shape,input_shape,3)))
        counter = 0
        while counter<deep:
            cnn.add(keras.layers.Conv2D(200, (5,5)))
            cnn.add((keras.layers.BatchNormalization()))
            cnn.add(keras.layers.Activation('relu'))
            cnn.add(keras.layers.MaxPool2D(pool_size=(3,3)))
            counter+=1
        cnn.add(keras.layers.Flatten())
        cnn.add(keras.layers.Dense(wide))
        cnn.add(keras.layers.BatchNormalization())
        cnn.add(keras.layers.Activation('relu'))
        cnn.add(keras.layers.Dense(wide/2))
        cnn.add(keras.layers.BatchNormalization())
        cnn.add(keras.layers.Activation('relu'))

        loss_function = 'categorical_crossentropy'
        activation = 'softmax'
        if n_classes == 2:
            loss_function = 'binary_crossentropy'
            activation = 'sigmoid'
        cnn.add(keras.layers.Dense(n_classes, activation=keras.layers.Activation(activation)))

        #masking_layer = keras.layers.Lambda(lambda x: x*mask)(bmlp.layers[-2].output)
        #if n_hidden_layers>1:
        #    while n_hidden_layers!=1:
        #        masking_layer= keras.layers.Dense(512, activation=keras.layers.Activation('sigmoid'))(masking_layer)
        #        n_hidden_layers-=1
        #decision_layer = keras.layers.Dense(10, activation=keras.layers.Activation('softmax'))(masking_layer)
        #masked_model = keras.models.Model(input= bmlp.input, output=decision_layer)
        model = keras.models.Model(input=cnn.input, output=cnn.output)
        model.compile(optimizer=optimizer,
                      loss=loss_function,
                      metrics=['accuracy'])
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.save_fold = save_fold

    def train(self, dataset):
        #import pdb; pdb.set_trace()
        x_train = dataset.x_train
        y_train = dataset.y_train
        x_train = x_train / 255.0
        x_train -= np.mean(x_train)
        np.random.seed(0)
        idxs_train = np.arange(len(x_train))
        np.random.shuffle(idxs_train)
        x_train = np.asarray(x_train[idxs_train])
        y_train = y_train[idxs_train]

        x_test = dataset.x_test
        y_test = dataset.y_test
        x_test = x_test / 255.0
        x_test -= np.mean(x_test)
        idxs_test = np.arange(len(x_test))
        np.random.shuffle(idxs_test)
        x_test = np.asarray(x_test[idxs_test])
        y_test = y_test[idxs_test]


        try:
            shape1, shape2 = y_train.shape()
        except:
            y_train = keras.utils.to_categorical(y_train, self.n_classes)
        try:
            shape1, shape2 = y_test.shape()
        except:
            y_test = keras.utils.to_categorical(y_test, self.n_classes)
        history=self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(x_test, y_test))
        self.training_history=history

    def train_and_compute_rcvs(self, dataset):

        x_train = dataset.x_train/255.
        x_train -= np.mean(x_train)
        y_train = dataset.y_train
        np.random.seed(0)
        idxs_train = np.arange(len(x_train))
        np.random.shuffle(idxs_train)
        x_train = np.asarray(x_train[idxs_train])
        y_train= np.asarray(y_train)
        y_train = y_train[idxs_train]

        try:
            shape1, shape2 = y_train.shape()
        except:
            y_train = keras.utils.to_categorical(y_train)

        history=[]
        embeddings=[]
        batch_size=self.batch_size
        print self.model.summary()
        
        #layers_of_interest = [layer.name for layer in self.model.layers[2:-1]]
        if self.deep==2:
            layer_idxs = [9,13,16]
        if self.deep==3:
             layer_idxs = [9,12,14]
        if self.deep==4:
             layer_idxs = [9,12,15,19,22]
        if self.deep==5:
             layer_idxs = [9,12,15,18,22,25]
            
        layers_of_interest = [self.model.layers[layer_idx].name for layer_idx in layer_idxs]
        print 'loi', layers_of_interest
        self.model.metrics_tensors += [layer.output for layer in self.model.layers if layer.name in layers_of_interest]
        epoch_number = 0

        
        n_batches = len(x_train)/self.batch_size
        remaining = len(x_train)-n_batches * self.batch_size
        while epoch_number <= self.epochs:
            print epoch_number
            batch_number = 0
            embedding_=[]

            for l in layers_of_interest:
                #print 'in layer ', l
                #print 'output shape ', self.model.get_layer(l).output.shape
                #print 'metrics tensors, ', self.model.metrics_tensors
                if len(self.model.get_layer(l).output.shape)<=2:
                    space = np.zeros((len(x_train), self.model.get_layer(l).output.shape[-1]))
                else:
                    x = self.model.get_layer(l).output.shape[-3]
                    y = self.model.get_layer(l).output.shape[-2]
                    z = self.model.get_layer(l).output.shape[-1]
                    space = np.zeros((len(x_train), x*y*z))

                embedding_.append(space)
            while batch_number <= n_batches:

                outs=self.model.train_on_batch(
                    x_train[batch_number*batch_size:batch_number*batch_size + batch_size], 
                    y_train[batch_number*batch_size:batch_number*batch_size + batch_size])
                embedding_[0][batch_number*batch_size: batch_number*batch_size+batch_size]= outs[2].reshape((min(batch_size,len(outs[2])),-1))
                embedding_[1][batch_number*batch_size: batch_number*batch_size+batch_size]=outs[3].reshape((len(outs[3]),-1))
                embedding_[2][batch_number*batch_size: batch_number*batch_size+batch_size]=outs[4].reshape((len(outs[4]),-1))
                #embedding_[3][batch_number*batch_size: batch_number*batch_size+batch_size]=outs[5].reshape((len(outs[5]),-1))

                history.append(outs[0])
                batch_number+=1
            #print self.save_fold
            source = self.save_fold#'/mnt/nas2/results/IntermediateResults/Mara/probes/imagenet/2H_lcp0.5'
            c=0
            if True:
                for l in layers_of_interest:
                    if 'max_pooling' in l:
                        tosave_= np.mean(embedding_[c].reshape(12775, 23*23,200), axis=1)
                        np.save('{}/imagenet_training_emb_e{}_l{}'.format(source,epoch_number, l), tosave_)
                    else:
                        np.save('{}/imagenet_training_emb_e{}_l{}'.format(source,epoch_number, l), embedding_[c])
                    c+=1
            del embedding_
            epoch_number +=1
        self.training_history=history
        self.embeddings = embeddings
        
    def _custom_eval(self, x, y, batch_size):
        ## correcting shape-related issues
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3], x.shape[4])
        y = y.reshape(y.shape[0],-1)
        #
        scores = []
        val_batch_no = 0
        start_batch = val_batch_no
        end_batch = start_batch + batch_size
        tot_batches = len(y) / batch_size
        # looping over data
        while val_batch_no < tot_batches:
            score = self.model.test_on_batch(x[start_batch:end_batch, :299, :299, :3],
                                             y[start_batch:end_batch])
            scores.append(score[1])
            val_batch_no += 1
            start_batch = end_batch
            end_batch += batch_size
        #print("Val: {}".format(np.mean(np.asarray(scores))))
        return np.mean(np.asarray(scores))
        
    def train_and_monitor_with_rcvs(self, dataset, layers_of_interest=[], directory_save='',custom_epochs=0):
        '''
        Train and Monitor with RCVs
        \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
          Similar to train and compute RCVs, we just keep track 
          of accuracy, partial accuracy (split in true and false labels)
          and we keep track of the embeddings corresponding to true and 
          false labels. 
          The function saves the embeddings at each epoch in a npy file.
          The mask of corrupted labels is saved in a separated npy file. 
        \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        Inputs:  
        dataset: Object of class either MNISTRandom, ImageNet10Random or Cifar10Random
          gives the object with rhe training data (dataset.x_train, dataset.y_train)
        name, string
          says the dataset name and if the dataet is corrupted with label corruption (lcp)
          or gaussian noise in the inputs (gnf). For example, if the datset is imagenet and we want to specify
          the respective lcp (x.xlcp_), or gnf values (x.xgnf_)
          we write datasetname_x.x followed by the name of the corruption, f.e.
          imagenet_0.8lcp_ for label corruption with probability 0.8
          or imagenet_0.5gnf_ for gaussian noise fraction of 0.5
        layers_of_interest, list
          allows to specify which layers we want to extract the embeddings from.
          ex.[6,11,14]
        '''
        directory_save = self.save_fold
        # train data with the original orderng (not shuffled yet)
        x_train = dataset.x_train/255.
        x_train -= np.mean(x_train)
        y_train = dataset.y_train
        # validation data with original orderng (not shuffled yet)
        x_val = np.asarray(dataset.x_test, dtype=np.float64)
        x_val -= np.mean(x_val)
        y_val = dataset.y_test
        # setting the seed for random
        try:
            np.random.seed(dataset.seed)
        except:
            np.random.seed(0)
        # mask of bool values set to true if the corresponding datapoint
        # was corrupted
        train_mask = dataset.train_mask
        # We shuffle the dataset indeces in a new array
        idxs_train = np.arange(len(x_train))
        np.random.shuffle(idxs_train)
        # List of corrupted and uncorrupted indeces in 
        # the original ordering of the data
        corrupted_idxs = np.argwhere(train_mask == True)
        uncorrupted_idxs = np.argwhere(train_mask == False)
        try:
            #import pdb; pdb.set_trace()
            np.save('{}/corrupted_idxs.npy'.format(directory_save), corrupted_idxs)
        except:
            print("ERROR saving corr idxs")
        try:
            np.save('{}/uncorrupted_idxs.npy'.format(directory_save), uncorrupted_idxs)
        except:
            print("ERROR saving uncorr idxs")
        ## x_train and y_train contain the data with the new shuffling
        orig_x_train=x_train
        orig_y_train=y_train
        x_train = np.asarray(x_train[idxs_train])
        y_train = y_train[idxs_train]
        #y_train = dataset.y_train
        #if x_train
        #x_train = x_train / 255.0
        # converting the labels to categorical 
        try:
            shape1, shape2 = y_train.shape()
        except:
            y_train = keras.utils.to_categorical(y_train)
        # converting also the original labels to categorical 
        # (for custom_eval_)
        try:
            shape1, shape2 = orig_y_train.shape()
        except:
            orig_y_train = keras.utils.to_categorical(orig_y_train)
        # variables for logs and monitoring
        history=[]
        embeddings=[]
        batch_size=self.batch_size
        print self.model.summary()
        ##### NOTE: the loi change from model to model
        layers_of_interest = [self.model.layers[layer_idx].name for layer_idx in layers_of_interest]
        print 'loi', layers_of_interest
        self.model.metrics_tensors += [layer.output for layer in self.model.layers if layer.name in layers_of_interest]
        epoch_number = 0
        n_batches = len(x_train)/self.batch_size
        remaining = len(x_train)-n_batches * self.batch_size
        while epoch_number <= self.epochs:
            print epoch_number
            batch_number = 0
            embedding_=[]
            for l in layers_of_interest:
                #print 'in layer ', l
                #print 'output shape ', self.model.get_layer(l).output.shape
                #print 'metrics tensors, ', self.model.metrics_tensors
                if len(self.model.get_layer(l).output.shape)<=2:
                    space = np.zeros((len(x_train), self.model.get_layer(l).output.shape[-1]))
                else:
                    x = self.model.get_layer(l).output.shape[-3]
                    y = self.model.get_layer(l).output.shape[-2]
                    z = self.model.get_layer(l).output.shape[-1]
                    space = np.zeros((len(x_train), x*y*z))

                embedding_.append(space)
            while batch_number <= n_batches:
                outs=self.model.train_on_batch(
                    x_train[batch_number*batch_size:batch_number*batch_size + batch_size],
                    y_train[batch_number*batch_size:batch_number*batch_size + batch_size])
                #import pdb; pdb.set_trace()
                embedding_[0][batch_number*batch_size: batch_number*batch_size+batch_size]=outs[2].reshape((min(batch_size,len(outs[2])),-1))
                embedding_[1][batch_number*batch_size: batch_number*batch_size+batch_size]=outs[3].reshape((len(outs[3]),-1))
                #embedding_[2][batch_number*batch_size: batch_number*batch_size+batch_size]=outs[4].reshape((len(outs[4]),-1))
                #embedding_[3][batch_number*batch_size: batch_number*batch_size+batch_size]=outs[5].reshape((len(outs[5]),-1))
                #print outs, outs
                history.append(outs[0])
                batch_number+=1
            c=0
            for l in layers_of_interest:
                np.save('{}/_training_emb_e{}_l{}'.format(directory_save,epoch_number, l), embedding_[c])
                c+=1
            del embedding_
            # here we check the partial accuracy
            if epoch_number %10 == 0:
                corrupted_acc = self._custom_eval(orig_x_train[corrupted_idxs],
                                                  orig_y_train[corrupted_idxs],
                                                  batch_size
                                                 )
                if len(uncorrupted_idxs>0):
                    uncorrupted_acc = self._custom_eval(orig_x_train[uncorrupted_idxs],
                                                      orig_y_train[uncorrupted_idxs],
                                                      batch_size
                                                     )
                    try:
                        with open(directory_save+'/uncorr_acc.txt', 'a') as log_file:
                            log_file.write("{}, ".format(uncorrupted_acc))
                    except:
                        log_file = open(directory_save+'/uncorr_acc.txt', 'w')
                        log_file.write("{}, ".format(uncorrupted_acc))
                try:
                    with open(directory_save+'/corr_acc.txt', 'a') as log_file:
                        log_file.write("{}, ".format(corrupted_acc))
                except:
                        log_file = open(directory_save+'/corr_acc.txt', 'w')
                        log_file.write("{}, ".format(corrupted_acc))

               
                
            epoch_number +=1
        self.training_history=history
        self.embeddings = embeddings

        
    def save(self, name, folder):
        try:
            os.listdir(folder)
        except:
            os.mkdir(folder)

        #model_json = self.model.to_json()
        #with open(folder+"/"+name+".json", "w") as json_file:
        #    json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights(folder+"/"+name+".h5")
        print("Saved model to disk")
        np.save(folder+'/'+name+'_history', self.training_history.history)

class CNNImagenet():
    '''
    ### NOTE: architecture modified to do Imagenet well
    Convolutional Neural Network for experiments on CIFAR
    input, crop(2,2),
    conv(200,5,5), bn, relu, maxpool(3,3),
    conv(200,5,5), bn, relu, maxpool(3,3),
    dense(384), bn, relu,
    dense(192), bn, relu,
    dense(n_classes), softmax

    Params
    deep: int (how many convolution blocks)
      Default 2
    wide: int (how many neurons in the first dense connection)
      Default 512
    optimizer: string
      Default SGD
    lr: float
      Default 1e-2
    epochs: int
      Default 10
    batch_size: int
      Default: 32
    input_shape: int
      Default 32
    n_classes: int
      Default 10
    '''

    def __init__(self, deep=2, wide=384, optimizer='SGD', lr=1e-2, epochs=9,
                 batch_size=14, input_shape=299, n_classes=10, **kwargs):

        #mask_shape = np.ones((1,512))
        #mask = keras.backend.variable(mask_shape)

        cnn = keras.models.Sequential()
        cnn.add(keras.layers.Cropping2D(cropping=((36,36),(36,36)), input_shape=(299,299,3)))
        counter = 0
        while counter<deep:
            cnn.add(keras.layers.Conv2D(200, (5,5)))
            cnn.add((keras.layers.BatchNormalization()))
            cnn.add(keras.layers.Activation('relu'))
            if counter<2:
                cnn.add(keras.layers.MaxPool2D(pool_size=(3,3)))
            counter+=1
        cnn.add(keras.layers.GlobalAveragePooling2D())
        #cnn.add(keras.layers.Flatten())
        cnn.add(keras.layers.Dense(wide))
        cnn.add(keras.layers.BatchNormalization())
        cnn.add(keras.layers.Activation('relu'))
        cnn.add(keras.layers.Dense(wide/2))
        cnn.add(keras.layers.BatchNormalization())
        cnn.add(keras.layers.Activation('relu'))

        loss_function = 'categorical_crossentropy'
        activation = 'softmax'
        if n_classes == 2:
            loss_function = 'binary_crossentropy'
            activation = 'sigmoid'
        cnn.add(keras.layers.Dense(n_classes, activation=keras.layers.Activation(activation)))

        #masking_layer = keras.layers.Lambda(lambda x: x*mask)(bmlp.layers[-2].output)
        #if n_hidden_layers>1:
        #    while n_hidden_layers!=1:
        #        masking_layer= keras.layers.Dense(512, activation=keras.layers.Activation('sigmoid'))(masking_layer)
        #        n_hidden_layers-=1
        #decision_layer = keras.layers.Dense(10, activation=keras.layers.Activation('softmax'))(masking_layer)
        #masked_model = keras.models.Model(input= bmlp.input, output=decision_layer)
        model = keras.models.Model(input=cnn.input, output=cnn.output)
        model.compile(optimizer=optimizer,
                      loss=loss_function,
                      metrics=['accuracy'])
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.deep=deep


    def train(self, dataset):
        #import pdb; pdb.set_trace()
        x_train = dataset.x_train
        y_train = dataset.y_train
        x_train = x_train / 255.0
        x_train -= np.mean(x_train)
        np.random.seed(0)
        idxs_train = np.arange(len(x_train))
        np.random.shuffle(idxs_train)
        x_train = np.asarray(x_train[idxs_train])
        y_train = y_train[idxs_train]

        x_test = dataset.x_test
        y_test = dataset.y_test
        x_test = x_test / 255.0
        x_test -= np.mean(x_test)
        idxs_test = np.arange(len(x_test))
        np.random.shuffle(idxs_test)
        x_test = np.asarray(x_test[idxs_test])
        y_test = y_test[idxs_test]


        try:
            shape1, shape2 = y_train.shape()
        except:
            y_train = keras.utils.to_categorical(y_train, self.n_classes)
        try:
            shape1, shape2 = y_test.shape()
        except:
            y_test = keras.utils.to_categorical(y_test, self.n_classes)
        history=self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(x_test, y_test))
        self.training_history=history

    def train_and_compute_rcvs(self, dataset):
        #import pdb; pdb.set_trace()

        x_train = dataset.x_train/255.
        x_train -= np.mean(x_train)
        y_train = dataset.y_train
        np.random.seed(0)
        idxs_train = np.arange(len(x_train))
        np.random.shuffle(idxs_train)
        x_train = np.asarray(x_train[idxs_train])
        y_train= np.asarray(y_train)
        y_train = y_train[idxs_train]
        #if x_train
        #x_train = x_train / 255.0

        try:
            shape1, shape2 = y_train.shape()
        except:
            y_train = keras.utils.to_categorical(y_train)

        history=[]
        embeddings=[]
        batch_size=self.batch_size
        print self.model.summary()

        #import pdb; pdb.set_trace()
        #layers_of_interest = [layer.name for layer in self.model.layers[2:-1]]
        if self.deep==2:
            layer_idxs = [9,13,16]
        if self.deep==3:
             layer_idxs = [9,12,14]
        if self.deep==4:
             layer_idxs = [9,12,15,19,22]
        if self.deep==5:
             layer_idxs = [9,12,15,18,22,25]

        layers_of_interest = [self.model.layers[layer_idx].name for layer_idx in layer_idxs]
        print 'loi', layers_of_interest
        self.model.metrics_tensors += [layer.output for layer in self.model.layers if layer.name in layers_of_interest]
        epoch_number = 0


        n_batches = len(x_train)/self.batch_size
        remaining = len(x_train)-n_batches * self.batch_size
        #if epoch_number > 1:
        while epoch_number <= self.epochs:
            print epoch_number
            batch_number = 0
            embedding_=[]

            for l in layers_of_interest:
                print 'in layer ', l
                print 'output shape ', self.model.get_layer(l).output.shape
                print 'metrics tensors, ', self.model.metrics_tensors
                if len(self.model.get_layer(l).output.shape)<=2:
                    space = np.zeros((len(x_train), self.model.get_layer(l).output.shape[-1]))
                else:
                    x = self.model.get_layer(l).output.shape[-3]
                    y = self.model.get_layer(l).output.shape[-2]
                    z = self.model.get_layer(l).output.shape[-1]
                    space = np.zeros((len(x_train), x*y*z))

                embedding_.append(space)
            while batch_number <= n_batches:

                outs=self.model.train_on_batch(
                    x_train[batch_number*batch_size:batch_number*batch_size + batch_size],
                    y_train[batch_number*batch_size:batch_number*batch_size + batch_size])
                #import pdb;pdb.set_trace()
                #print out[0]
                #import pdb; pdb.set_trace()
                embedding_[0][batch_number*batch_size: batch_number*batch_size+batch_size]= outs[2].reshape((min(batch_size,len(outs[2])),-1))
                embedding_[1][batch_number*batch_size: batch_number*batch_size+batch_size]=outs[3].reshape((len(outs[3]),-1))
                embedding_[2][batch_number*batch_size: batch_number*batch_size+batch_size]=outs[4].reshape((len(outs[4]),-1))
                #embedding_[3][batch_number*batch_size: batch_number*batch_size+batch_size]=outs[5].reshape((len(outs[5]),-1))

                #print outs, outs
                history.append(outs[0])
                batch_number+=1
            #import pdb; pdb.set_trace()
            source = '/mnt/nas2/results/IntermediateResults/Mara/probes/imagenet/2H_lcp0.4'
            c=0
            if True:
                for l in layers_of_interest:
                    if 'max_pooling' in l:
                        #import pdb; pdb.set_trace()
                        tosave_= np.mean(embedding_[c].reshape(12775, 23*23,200), axis=1)
                        np.save('{}/imagenet_training_emb_e{}_l{}'.format(source,epoch_number, l), tosave_)
                        #np.mean(embedding_[0].reshape(12775, 31*31,200), axis=1).shape
                    else:
                        np.save('{}/imagenet_training_emb_e{}_l{}'.format(source,epoch_number, l), embedding_[c])
                    c+=1
            del embedding_
            #embeddings.append(embedding_)
            epoch_number +=1
        self.training_history=history
        self.embeddings = embeddings


    def save(self, name, folder):
        try:
            os.listdir(folder)
        except:
            os.mkdir(folder)

        #model_json = self.model.to_json()
        #with open(folder+"/"+name+".json", "w") as json_file:
        #    json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights(folder+"/"+name+".h5")
        print("Saved model to disk")
        np.save(folder+'/'+name+'_history', self.training_history.history)

'''
#### Old, used for CIFAR
class CNN():

    Convolutional Neural Network for experiments on CIFAR
    input, crop(2,2),
    conv(200,5,5), bn, relu, maxpool(3,3),
    conv(200,5,5), bn, relu, maxpool(3,3),
    dense(384), bn, relu,
    dense(192), bn, relu,
    dense(n_classes), softmax

    Params
    deep: int (how many convolution blocks)
      Default 2
    wide: int (how many neurons in the first dense connection)
      Default 512
    optimizer: string
      Default SGD
    lr: float
      Default 1e-2
    epochs: int
      Default 10
    batch_size: int
      Default: 32
    input_shape: int
      Default 32
    n_classes: int
      Default 10


    def __init__(self, deep=2, wide=384, optimizer='SGD', lr=1e-2, epochs=10,
                 batch_size=32, input_shape=32, n_classes=10, **kwargs):

        #mask_shape = np.ones((1,512))
        #mask = keras.backend.variable(mask_shape)

        cnn = keras.models.Sequential()
        cnn.add(keras.layers.Cropping2D(cropping=((2,2),(2,2)), input_shape=(32,32,3)))
        counter = 0
        while counter<deep:
            cnn.add(keras.layers.Conv2D(200, (5,5)))
            cnn.add((keras.layers.BatchNormalization()))
            cnn.add(keras.layers.Activation('relu'))
            cnn.add(keras.layers.MaxPool2D(pool_size=(3,3)))
            counter+=1
        cnn.add(keras.layers.Flatten())
        cnn.add(keras.layers.Dense(wide))
        cnn.add(keras.layers.BatchNormalization())
        cnn.add(keras.layers.Activation('relu'))
        cnn.add(keras.layers.Dense(wide/2))
        cnn.add(keras.layers.BatchNormalization())
        cnn.add(keras.layers.Activation('relu'))

        loss_function = 'categorical_crossentropy'
        activation = 'softmax'
        if n_classes == 2:
            loss_function = 'binary_crossentropy'
            activation = 'sigmoid'
        cnn.add(keras.layers.Dense(n_classes, activation=keras.layers.Activation(activation)))

        #masking_layer = keras.layers.Lambda(lambda x: x*mask)(bmlp.layers[-2].output)
        #if n_hidden_layers>1:
        #    while n_hidden_layers!=1:
        #        masking_layer= keras.layers.Dense(512, activation=keras.layers.Activation('sigmoid'))(masking_layer)
        #        n_hidden_layers-=1
        #decision_layer = keras.layers.Dense(10, activation=keras.layers.Activation('softmax'))(masking_layer)
        #masked_model = keras.models.Model(input= bmlp.input, output=decision_layer)
        model = keras.models.Model(input=cnn.input, output=cnn.output)
        model.compile(optimizer=optimizer,
                      loss=loss_function,
                      metrics=['accuracy'])
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_classes = n_classes


    def train(self, dataset):
        #import pdb; pdb.set_trace()
        x_train = dataset.x_train
        y_train = dataset.y_train
        x_train = x_train / 255.0

        try:
            shape1, shape2 = y_train.shape()
        except:
            y_train = keras.utils.to_categorical(y_train, self.n_classes)
        history=self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_slit=0.2)
        self.training_history=history

    def save(self, name, folder):
        try:
            os.listdir(folder)
        except:
            os.mkdir(folder)

        #model_json = self.model.to_json()
        #with open(folder+"/"+name+".json", "w") as json_file:
        #    json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(folder+"/"+name+".h5")
        print("Saved model to disk")
        np.save(folder+'/'+name+'_history', self.training_history.history)
'''

class InceptionV3():
    '''
    InceptionV3 architechture with options
    to learn RCVs over training, or to instanciate a
    flawed network

    Input Parameters

    optimizer: string, default Adam
    lr: float, default 0.01
    beta_1: float, default 0.9
    beta_2: float, default 0.999
    epochs: int, default 1000
    batch_size: int, default 32
    input_shape: int, default 299
    n_classes: int, default 47
    '''

    def __init__(self, optimizer='Adam', lr=0.01, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0.0, amsgrad=False, epochs=1000,
                 batch_size=32, input_shape=299, n_classes=47, **kwargs):

        model = keras.applications.inception_v3.InceptionV3(include_top=True,
                                                          weights=None,#'imagenet', #None,
                                                          input_tensor=None,
                                                          input_shape=(input_shape,input_shape,3),
                                                          pooling=None,
                                                          classes=n_classes)

        model.compile(
            keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_1, epsilon=epsilon, decay=decay, amsgrad=amsgrad),
            loss = keras.losses.categorical_crossentropy,
                    metrics=['acc']
                   )
        '''
        model.compile(
            keras.optimizers.SGD(lr=lr, momentum=0.9, decay=decay, nesterov=True),
            loss = keras.losses.categorical_crossentropy,
                    metrics=['acc']
                   )
        '''
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_classes = n_classes

    def get_activations(self, inputs, layer):

        get_layer_output = K.function([self.model.layers[0].input],
                              [self.model.get_layer(layer).output])
        feats = get_layer_output([inputs])
        return feats[0]

    def train(self, dataset, custom_epochs=0):
        '''
        Trains the CNN on the dataset
        the input dataset is a variable with the fields
        dataset.x_train, dataset.y_train, dataset.x_val, dataset.y_val


        '''

        x_train = dataset.x_train
        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = dataset.y_train

        x_val = dataset.x_val
        x_val = np.asarray(x_val, dtype=np.float32)
        y_val = dataset.y_val

        # We first need to shuffle the dataset indexes then we launch training
        shuffle_idxs_train = np.arange(len(x_train))
        np.random.shuffle(shuffle_idxs_train)
        shuffle_idxs_val = np.arange(len(x_val))
        np.random.shuffle(shuffle_idxs_val)

        # We also first clean the data
        x_train[:,:,:,:3] -= np.mean(x_train[:,:,:,:3])
        x_train[:,:,:,:3] /= np.std(x_train[:,:,:,:3])
        x_val -= np.mean(x_val[:,:,:,:3])
        x_val /= np.std(x_val[:,:,:,:3])

        try:
            shape1, shape2 = y_train.shape()
        except:
            y_train = keras.utils.to_categorical(y_train, self.n_classes)

        try:
            shape1, shape2 = y_val.shape()
        except:
            y_val = keras.utils.to_categorical(y_val, self.n_classes)

        # We instanciate an Image data generator to extract random croppings of size 299x299x3
        # from the images in x_train
        datagen = ImageDataGenerator(featurewise_center=False,
                                    featurewise_std_normalization=False,
                                    random_cropping=True
                                    )
        datagen.fit(x_train[shuffle_idxs_train, :, :, :3])
        train_generator = datagen.flow(x_train[shuffle_idxs_train],
                                      y_train[shuffle_idxs_train],
                                      batch_size=self.batch_size
                                      )
        if custom_epochs>0:
            epochs = custom_epochs
        else:
            epochs = self.epochs

        history = self.model.fit_generator(train_generator,
                                           steps_per_epoch= len(x_train)/self.batch_size,
                                           epochs=epochs,
                                           validation_data=(x_val[shuffle_idxs_val],
                                                            y_val[shuffle_idxs_val]),
                                           )

        #history=self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_slit=0.2)
        self.training_history=history

    def train_and_compute_rcvs(self, dataset, layers_of_interest=[], custom_epochs=0):
        import time
        # making the data ready for training
        x_train = dataset.x_train
        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = dataset.y_train

        x_val = dataset.x_val
        x_val = np.asarray(x_val, dtype=np.float32)
        y_val = dataset.y_val

        # We first need to shuffle the dataset indexes then we launch training
        shuffle_idxs_train = np.arange(len(x_train))
        np.random.shuffle(shuffle_idxs_train)
        try:
            localtime = time.localtime(time.time())
            os.mkdir('/mnt/nas2/results/IntermediateResults/Mara/probes/experiment_{}.{}_{}.{}'.format(localtime.tm_mday, localtime.tm_mon, localtime.tm_hour, localtime.tm_min))
            directory_save = '/mnt/nas2/results/IntermediateResults/Mara/probes/experiment_{}.{}_{}.{}'.format(localtime.tm_mday, localtime.tm_mon, localtime.tm_hour, localtime.tm_min)
        except:
            print("ERR has occurred")
        np.save(directory_save+'/shuffle_idxs_training', shuffle_idxs_train)
        shuffle_idxs_val = np.arange(len(x_val))
        np.random.shuffle(shuffle_idxs_val)
        np.save(directory_save+'/shuffle_idxs_validation', shuffle_idxs_val)

        # We clean the data
        x_train[:,:,:,:3] -= np.mean(x_train[:,:,:,:3])
        x_train[:,:,:,:3] /= np.std(x_train[:,:,:,:3])
        x_val -= np.mean(x_val[:,:,:,:3])
        x_val /= np.std(x_val[:,:,:,:3])
        #import pdb; pdb.set_trace()
        # At this point in time the mean and std of the image data are resp 0 and 1
        # The mask of the image boundaries is left untouched (checked via pdb)
        try:
            shape1, shape2 = y_train.shape()
        except:
            y_train = keras.utils.to_categorical(y_train, self.n_classes)

        try:
            shape1, shape2 = y_val.shape()
        except:
            y_val = keras.utils.to_categorical(y_val, self.n_classes)

        '''
        try:
            shape1, shape2 = y_train.shape()
        except:
            y_train = keras.utils.to_categorical(y_train, self.n_classes)
        '''
        history=[]
        #embeddings=[]
        #layers_of_interest = [self.model.layers[layer_idx].name for layer_idx in [2,6,11,14]]
        # getting the embeddings at the layers of interest
        #self.model.metrics_tensors += [layer.output for layer in self.model.layers if layer.name in layers_of_interest]
        st = time.time()
        # We instanciate an Image data generator to extract random croppings of size 299x299x3
        # from the images in x_train

        datagen = ImageDataGenerator(featurewise_center=False,
                                    featurewise_std_normalization=False,
                                    random_cropping=True
                                    )

        datagen.fit(x_train[shuffle_idxs_train, :, :, :3])
        #train_generator = datagen.flow(x_train[shuffle_idxs_train],
        #                              y_train[shuffle_idxs_train],
        #                              batch_size=self.batch_size
        #                              )
        print('Train generator ready, time elapsed: {}'.format(time.time()-st))
        if custom_epochs>0:
            epochs = custom_epochs
        else:
            epochs = self.epochs

        batch_size=self.batch_size
        tot_val_batches = len(x_val)/batch_size

        epoch_number = 0

        while epoch_number <= self.epochs:
            #print(epoch_number)
            batch_number = 0
            start_batch=0
            end_batch = batch_size
            tr_losses = []
            tr_accs = []

            '''
            while batch_number<len(x_train)/batch_size:
                tr_loss, acc = self.model.train_on_batch(x_train[shuffle_idxs_train[start_batch:end_batch], :299, :299, :3],
                                                         y_train[shuffle_idxs_train[start_batch:end_batch]]
                                                      )
                tr_losses.append(tr_loss)
                tr_accs.append(acc)
                batch_number +=1
            #print(tr_accs)
            '''
            '''
            if epoch_number % 10 == 0:
                #import pdb; pdb.set_trace()
                ## saving validation!At the end of the epoch we validate on the
                # validation split and save the embeddings


                scores = []
                val_batch_no = 0
                start_batch = val_batch_no
                end_batch = start_batch + batch_size
                while val_batch_no < tot_val_batches:
                    #index_start = shuffle_idxs_val[]

                    score = self.model.test_on_batch(x_val[shuffle_idxs_val[start_batch:end_batch]],
                                                     y_val[shuffle_idxs_val[start_batch:end_batch]])

                    #score = self.model.evaluate(x_val[val_batch_no*batch_size : val_batch_no*batch_size + batch_size],
                    #                               y_val[val_batch_no*batch_size : val_batch_no*batch_size + batch_size],
                    #                               batch_size=32)
                    #print("Validation batch no: {}, acc: {}".format(val_batch_no, score))
                    scores.append(score[1])
                    val_batch_no += 1
                    start_batch = end_batch
                    end_batch += batch_size
                #print(scores)
                print("Val: {}".format(np.mean(np.asarray(scores))))
            '''
            if epoch_number % 10 == 0:
                # allocating space and reusing the variable embedding_
                embedding_=[]
                for l in layers_of_interest:
                    if len(self.model.get_layer(l).output.shape)<=2:
                        space = np.zeros((len(x_val), self.model.get_layer(l).output.shape[-1]))
                    else:
                         #x = self.model.get_layer(l).output.shape[-3]
                        #y = self.model.get_layer(l).output.shape[-2]
                        z = self.model.get_layer(l).output.shape[-1]
                        #space = np.zeros((len(x_val), x,y,z))
                        # Update: only storing/saving the pooled embeddings bc I'm filling up all the hard drives
                        space = np.zeros((len(x_val), z))
                    embedding_.append(space)
                # this is a layer counter. I use it only to store the activations in the correct place in embedding_
                k=0
                for l in layers_of_interest:
                    val_batch_no = 0
                    start_batch = val_batch_no
                    end_batch = start_batch + batch_size
                    while val_batch_no <= tot_val_batches:
                        outs=self.get_activations(x_val[shuffle_idxs_val[start_batch:end_batch]], l)
                        # Global Average Pooling the activations : saves space and removes pixel dependencies
                        dims = outs.shape
                        avgp_outs = skimage.measure.block_reduce(outs, (1, dims[1], dims[2],1), np.mean)
                        avgp_outs= avgp_outs.reshape((dims[0],-1))
                        #import pdb; pdb.set_trace()
                        embedding_[k][start_batch:end_batch]=avgp_outs
                        ## end GAP. Like this it should take less space on the disk and solve all the problems
                        val_batch_no += 1
                        start_batch=end_batch
                        end_batch += batch_size
                    k += 1
                # Saving outputs on an external file
                c=0
                for l in layers_of_interest:
                    np.save(directory_save+'/_fix_training_emb_e{}_l{}_val_data'.format(epoch_number, l), embedding_[c])
                    # saving all the GAP acts with the name starting by _ so I can differentiate them
                    c+=1

            for x_batch, y_batch in datagen.flow(x_train[shuffle_idxs_train],
                                                 y_train[shuffle_idxs_train],
                                                 batch_size=batch_size):
                # training step
                tr_loss, acc = self.model.train_on_batch(x_batch, y_batch)
                tr_losses.append(tr_loss)
                tr_accs.append(acc)

                batch_number += 1
                # stopping condition if all data have been passed through
                # because the generator loops indefinitely
                if batch_number >= len(x_train)/batch_size:
                    break
            print('Epoch: {}, loss: {}, acc: {}'.format(epoch_number,
                                                np.mean(np.asarray(tr_losses)),
                                                np.mean(np.asarray(tr_accs))
                                               ))
        epoch_number +=1
        self.training_history=history
        #self.embeddings = embeddings
    def _custom_eval(self, x, y, batch_size):
        ## correcting shape-related issues
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3], x.shape[4])
        y = y.reshape(y.shape[0],-1)
        #
        scores = []
        val_batch_no = 0
        start_batch = val_batch_no
        end_batch = start_batch + batch_size
        tot_batches = len(y) / batch_size
        # looping over data
        while val_batch_no < tot_batches:
            score = self.model.test_on_batch(x[start_batch:end_batch, :299, :299, :3],
                                             y[start_batch:end_batch])
            scores.append(score[1])
            val_batch_no += 1
            start_batch = end_batch
            end_batch += batch_size
        #print("Val: {}".format(np.mean(np.asarray(scores))))
        return np.mean(np.asarray(scores))

    def train_and_monitor_with_rcvs(self, dataset, layers_of_interest=[], custom_epochs=0):
        import time
        # making the data ready for training
        x_train = dataset.x_train
        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = dataset.y_train

        x_val = dataset.x_val
        x_val = np.asarray(x_val, dtype=np.float32)
        y_val = dataset.y_val

        train_mask = dataset.train_mask

        # We first need to shuffle the dataset indexes then we launch training
        shuffle_idxs_train = np.arange(len(x_train))
        np.random.shuffle(shuffle_idxs_train)

        corrupted_idxs = np.argwhere(train_mask == True)
        uncorrupted_idxs = np.argwhere(train_mask == False)
        #import pdb; pdb.set_trace()

        try:
            localtime = time.localtime(time.time())
            os.mkdir('/mnt/nas2/results/IntermediateResults/Mara/probes/experiment_{}.{}_{}.{}'.format(localtime.tm_mday, localtime.tm_mon, localtime.tm_hour, localtime.tm_min))
            directory_save = '/mnt/nas2/results/IntermediateResults/Mara/probes/experiment_{}.{}_{}.{}'.format(localtime.tm_mday, localtime.tm_mon, localtime.tm_hour, localtime.tm_min)
        except:
            print("ERR has occurred")
        np.save(directory_save+'/shuffle_idxs_training', shuffle_idxs_train)
        shuffle_idxs_val = np.arange(len(x_val))
        np.random.shuffle(shuffle_idxs_val)
        np.save(directory_save+'/shuffle_idxs_validation', shuffle_idxs_val)

        # We clean the data
        x_train[:,:,:,:3] -= np.mean(x_train[:,:,:,:3])
        x_train[:,:,:,:3] /= np.std(x_train[:,:,:,:3])
        x_val -= np.mean(x_val[:,:,:,:3])
        x_val /= np.std(x_val[:,:,:,:3])
        #import pdb; pdb.set_trace()
        # At this point in time the mean and std of the image data are resp 0 and 1
        # The mask of the image boundaries is left untouched (checked via pdb)
        try:
            shape1, shape2 = y_train.shape()
        except:
            y_train = keras.utils.to_categorical(y_train, self.n_classes)

        try:
            shape1, shape2 = y_val.shape()
        except:
            y_val = keras.utils.to_categorical(y_val, self.n_classes)

        '''
        try:
            shape1, shape2 = y_train.shape()
        except:
            y_train = keras.utils.to_categorical(y_train, self.n_classes)
        '''
        history=[]
        #embeddings=[]
        #layers_of_interest = [self.model.layers[layer_idx].name for layer_idx in [2,6,11,14]]
        # getting the embeddings at the layers of interest
        #self.model.metrics_tensors += [layer.output for layer in self.model.layers if layer.name in layers_of_interest]
        st = time.time()
        # We instanciate an Image data generator to extract random croppings of size 299x299x3
        # from the images in x_train
        datagen = ImageDataGenerator(featurewise_center=False,
                                    featurewise_std_normalization=False,
                                    random_cropping=True
                                    )
        datagen.fit(x_train[shuffle_idxs_train, :, :, :3])
        train_generator = datagen.flow(x_train[shuffle_idxs_train],
                                      y_train[shuffle_idxs_train],
                                      batch_size=self.batch_size
                                      )
        print('Train generator ready, time elapsed: {}'.format(time.time()-st))
        if custom_epochs>0:
            epochs = custom_epochs
        else:
            epochs = self.epochs

        batch_size=self.batch_size
        tot_val_batches = len(x_val)/batch_size

        epoch_number = 0

        while epoch_number <= self.epochs:
            #print(epoch_number)
            batch_number = 0
            start_batch=0
            end_batch = batch_size
            tr_losses = []
            tr_accs = []

            for x_batch, y_batch in datagen.flow(x_train[shuffle_idxs_train],
                                                 y_train[shuffle_idxs_train],
                                                 batch_size=batch_size):

                # training step
                tr_loss, acc = self.model.train_on_batch(x_batch, y_batch)
                tr_losses.append(tr_loss)
                tr_accs.append(acc)

                batch_number += 1
                # stopping condition if all data have been passed through
                # because the generator loops indefinitely
                if batch_number >= len(x_train)/batch_size:
                    break
            if len(uncorrupted_idxs)>0:
                if epoch_number %10 == 0:
                    '''storing the corr, uncorr acc every 10 epochs to save space/time'''
                    corrupted_acc = self._custom_eval(x_train[corrupted_idxs],
                                            y_train[corrupted_idxs],
                                            batch_size)
                    uncorrupted_acc = self._custom_eval(x_train[uncorrupted_idxs],
                                            y_train[uncorrupted_idxs],
                                            batch_size)
                    try:
                        with open(directory_save+'/corr_acc.txt', 'a') as log_file:
                            log_file.write("{}, ".format(corrupted_acc))
                    except:
                        log_file = open(directory_save+'/corr_acc.txt', 'w')
                        log_file.write("{}, ".format(corrupted_acc))

                    try:
                        with open(directory_save+'/uncorr_acc.txt', 'a') as log_file:
                            log_file.write("{}, ".format(uncorrupted_acc))
                    except:
                        log_file = open(directory_save+'/uncorr_acc.txt', 'w')
                        log_file.write("{}, ".format(uncorrupted_acc))
                    ### introduce _custom_eval(x_train[corrupted_idxs])
                    #### _custom_eval(x_train[uncorrupted_idxs])
            '''
            while batch_number<len(x_train)/batch_size:
                tr_loss, acc = self.model.train_on_batch(x_train[shuffle_idxs_train[start_batch:end_batch], :299, :299, :3],
                                                         y_train[shuffle_idxs_train[start_batch:end_batch]]
                                                      )
                tr_losses.append(tr_loss)
                tr_accs.append(acc)
                batch_number +=1
            #print(tr_accs)
            '''
            print('Epoch: {}, loss: {}, acc: {}'.format(epoch_number,
                                                        np.mean(np.asarray(tr_losses)),
                                                        np.mean(np.asarray(tr_accs))
                                                       )
                 )

            if epoch_number % 10 == 0:
                #import pdb; pdb.set_trace()
                ## saving validation!At the end of the epoch we validate on the
                # validation split and save the embeddings


                scores = []
                val_batch_no = 0
                start_batch = val_batch_no
                end_batch = start_batch + batch_size
                while val_batch_no < tot_val_batches:
                    #index_start = shuffle_idxs_val[]

                    score = self.model.test_on_batch(x_val[shuffle_idxs_val[start_batch:end_batch]],
                                                     y_val[shuffle_idxs_val[start_batch:end_batch]])

                    #score = self.model.evaluate(x_val[val_batch_no*batch_size : val_batch_no*batch_size + batch_size],
                    #                               y_val[val_batch_no*batch_size : val_batch_no*batch_size + batch_size],
                    #                               batch_size=32)
                    #print("Validation batch no: {}, acc: {}".format(val_batch_no, score))
                    scores.append(score[1])
                    val_batch_no += 1
                    start_batch = end_batch
                    end_batch += batch_size
                #print(scores)
                print("Val: {}".format(np.mean(np.asarray(scores))))
            '''
            if epoch_number % 1000 == 0:
                # allocating space and reusing the variable embedding_
                embedding_=[]
                for l in layers_of_interest:
                    if len(self.model.get_layer(l).output.shape)<=2:
                        space = np.zeros((len(x_train), self.model.get_layer(l).output.shape[-1]))
                    else:
                         #x = self.model.get_layer(l).output.shape[-3]
                        #y = self.model.get_layer(l).output.shape[-2]
                        z = self.model.get_layer(l).output.shape[-1]
                        #space = np.zeros((len(x_val), x,y,z))
                        # Update: only storing/saving the pooled embeddings bc I'm filling up all the hard drives
                        space = np.zeros((len(x_train), z))
                    embedding_.append(space)
                # this is a layer counter. I use it only to store the activations in the correct place in embedding_
                k=0
                for l in layers_of_interest:
                    val_batch_no = 0
                    start_batch = val_batch_no
                    end_batch = start_batch + batch_size
                    while val_batch_no <= tot_val_batches:
                        outs=self.get_activations(x_train[shuffle_idxs_val[start_batch:end_batch]], l)
                        # Global Average Pooling the activations : saves space and removes pixel dependencies
                        dims = outs.shape
                        avgp_outs = skimage.measure.block_reduce(outs, (1, dims[1], dims[2],1), np.mean)
                        avgp_outs= avgp_outs.reshape((dims[0],-1))
                        #import pdb; pdb.set_trace()
                        embedding_[k][start_batch:end_batch]=avgp_outs
                        ## end GAP. Like this it should take less space on the disk and solve all the problems
                        val_batch_no += 1
                        start_batch=end_batch
                        end_batch += batch_size
                    k += 1
                # Saving outputs on an external file
                c=0
                for l in layers_of_interest:
                    np.save(directory_save+'/_training_emb_e{}_l{}_val_data'.format(epoch_number, l), embedding_[c])
                    # saving all the GAP acts with the name starting by _ so I can differentiate them
                    c+=1

                    if epoch_number % 10 == 0:
                # allocating space and reusing the variable embedding_
                embedding_=[]
                for l in layers_of_interest:
                    if len(self.model.get_layer(l).output.shape)<=2:
                        space = np.zeros((len(x_train), self.model.get_layer(l).output.shape[-1]))
                    else:
                         #x = self.model.get_layer(l).output.shape[-3]
                        #y = self.model.get_layer(l).output.shape[-2]
                        z = self.model.get_layer(l).output.shape[-1]
                        #space = np.zeros((len(x_val), x,y,z))
                        # Update: only storing/saving the pooled embeddings bc I'm filling up all the hard drives
                        space = np.zeros((len(x_train), z))
                    embedding_.append(space)
                # this is a layer counter. I use it only to store the activations in the correct place in embedding_
                k=0
                for l in layers_of_interest:
                    val_batch_no = 0
                    start_batch = val_batch_no
                    end_batch = start_batch + batch_size
                    while val_batch_no <= tot_val_batches:
                        outs=self.get_activations(x_train[shuffle_idxs_val[start_batch:end_batch]], l)
                        # Global Average Pooling the activations : saves space and removes pixel dependencies
                        dims = outs.shape
                        avgp_outs = skimage.measure.block_reduce(outs, (1, dims[1], dims[2],1), np.mean)
                        avgp_outs= avgp_outs.reshape((dims[0],-1))
                        #import pdb; pdb.set_trace()
                        embedding_[k][start_batch:end_batch]=avgp_outs
                        ## end GAP. Like this it should take less space on the disk and solve all the problems
                        val_batch_no += 1
                        start_batch=end_batch
                        end_batch += batch_size
                    k += 1
                # Saving outputs on an external file
                c=0
                for l in layers_of_interest:
                    np.save(directory_save+'/_training_emb_e{}_l{}_val_data'.format(epoch_number, l), embedding_[c])
                    # saving all the GAP acts with the name starting by _ so I can differentiate them
                    c+=1
             '''
            epoch_number +=1
        self.training_history=history
        #self.embeddings = embeddings

    def save(self, name, folder):
        try:
            os.listdir(folder)
        except:
            os.mkdir(folder)

        #model_json = self.model.to_json()
        #with open(folder+"/"+name+".json", "w") as json_file:
        #    json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(folder+"/"+name+".h5")
        print("Saved model to disk")
        try:
            np.save(folder+'/'+name+'_history', self.training_history.history)
        except:
            print "History not saved"
        return
