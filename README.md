# Interpreting Intentionally Flawed Models with Linear Probes
Mara Graziani, Henning Muller and Vincent Andrearczyk

submitted to SDVLC at ICCV2019

## 1. Train models

- DTD training is in dtd_experiments.ipynb

'''python
#create model
inceptionv3 = InceptionV3(batch_size=64)
#load dataset: label_corrupt_p is the fraction of label corruption
dataset = Dataset(train_data, val_data, test_data, label_corrupt_p = 0.0, random_seed=0)
#model training and extraction of activations at layers_of_interest
inceptionv3.train_and_compute_rcvs(dataset, layers_of_interest=['mixed0', 'mixed2','mixed4', 'mixed6'])
#save model weights
inceptionv3.save('random_fix_0.0lcp_rep0', 'results/trained_models')
'''
