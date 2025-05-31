#!/usr/bin/python

import pickle
import gib_detect_train
print("ici 1")
model_data = pickle.load(open('gib_model.pki', 'rb'))

while True:
    print("ici 2")
    l = input()
    print("ici 3")
    model_mat = model_data['mat']
    print("ici 4")
    threshold = model_data['thresh']
    print("ici 5")
    print(gib_detect_train.avg_transition_prob(l, model_mat) > threshold)
