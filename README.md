# Interpreting Intentionally Flawed Models with Linear Probes
Mara Graziani, Henning Muller and Vincent Andrearczyk

submitted to SDVLC at ICCV2019

## 1. Train models

- shallowCNN on ImageNet10, see train_imgnt10.ipynb

```python
# saving folder:
SAVE_FOLD='.results/trained_models/'
#create model
cnn = CNN(deep=2)
#load dataset: see load_imagenet_from_nas.ipynb to create data-train and data-val splits
imagenet10=ImageNet10Random(classes=classes, path_to_train='./data-train.h5', path_to_val='./data-val.h5')
#model training
model.train(imagenet10)
#save model weights
model.save('imgnet_0.0lcp_rep0', SAVE_FOLD)
```

- InceptionV3 on DTD, see dtd_experiments.ipynb

```python
#create model
inceptionv3 = InceptionV3(batch_size=64)
#load dataset: label_corrupt_p is the fraction of label corruption
dataset = Dataset(train_data, val_data, test_data, label_corrupt_p = 0.0, random_seed=0)
#model training and extraction of activations at layers_of_interest
inceptionv3.train_and_compute_rcvs(dataset, layers_of_interest=['mixed0', 'mixed2','mixed4', 'mixed6'])
#save model weights
inceptionv3.save('dtd_0.0lcp_rep0', SAVE_FOLD)
```

- MLP on MNIST, see mnist_experiments.ipynb

```python
model = MLP(deep=6, wide=4096, epochs=1000)
mnist_data = MNISTRandom(label_corrupt_p=0.0)
model.train(mnist_data)
model.save('mnist_0.0lcp_rep0', SAVE_FOLD)
```

# 2. Linear probes improve over training 
```python
# we compute the RCV for a set of concepts 
# at different layers (layers_of_interest) and every 50 epochs during training
# we evaluate the rsquared and MSE of the regression
for e in range(0,last_epoch, 50):
    r2 = reg_at_epoch(e,exps_folder,0)
    for l in layers_of_interest:
        r2 = reg_at_epoch(e, l, exps_folder)
        for c in concepts:
            r2ss[c][l].append(r2[c])
# r2ss is a dictionary with concepts and layers as keys
# for a couple concept, layer there is a list of the r2 over the training epochs
