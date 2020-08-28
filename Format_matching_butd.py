import numpy as np
import json
import pickle as pkl
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet
import cv2
import os
from itertools import chain
from tqdm import tqdm

dicts_butd = pkl.load(open('butd_idx_label_pruned.pkl', 'rb'))


for i, img_idx in enumerate(tqdm(dicts_butd)):
    rel_list = []
    for i in range(len(dicts_butd[img_idx]['pred_attri'])):
        rel_list.append([i,i])
    dicts_butd[img_idx]['pred_rel_inds'] = rel_list
    dicts_butd[img_idx]['rel_inds'] = dicts_butd[img_idx]['pred_attri']
    del dicts_butd[img_idx]['pred_attri']
    for i in range(len(dicts_butd[img_idx]['pred_boxes'])):
        for j in range(len(dicts_butd[img_idx]['pred_boxes'][i])):
            dicts_butd[img_idx]['pred_boxes'][i][j] = float(dicts_butd[img_idx]['pred_boxes'][i][j])

pickle_out = open("butd_idx_label_pruned_addrel.pkl","wb")
pkl.dump(dicts_butd, pickle_out)
