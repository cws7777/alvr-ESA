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


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    if xB < xA or yB < yA:
        iou = 0.
    # compute the area of intersection rectangle

    interArea = (xB - xA + 1) * (yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou




def get_synset(obj):
    synset = []
    try:
        cb = wordnet.synsets(obj.replace(" ", "_"))[0]

        for item in cb.lemmas():  # add lemmas
            synset.append(item.name())
        for item in cb.hypernyms():
            for it in item.lemma_names():
                synset.append(it)
        for item in cb.hyponyms():
            for it in item.lemma_names():
                synset.append(it)
    except Exception as ex:
        pass
    return synset

def compare(sg, node_num, new_sg, i):

    base_obj = new_sg['nodes'][i]
    target_obj = sg['pred_classes'][node_num]
    #print('base: ',base_obj)
    #print('target: ',target_obj)
    synset_base = get_synset(base_obj)
    synset_target = get_synset(target_obj)

    if base_obj in synset_target or target_obj in synset_base:
        iou_value = bb_intersection_over_union(sg['pred_boxes'][node_num], new_sg['bboxes'][i])
        if iou_value >0.3:
            return True
    return False



def compare_and_add(sg, node_num, new_sg, final_idx, new_item):
    for i in range(len(new_sg['nodes'])):
        comp_result = compare(sg, node_num, new_sg, i)
        if comp_result == True:
            np_rel = np.array(sg['pred_rel_inds'])
            np_rel[np_rel==node_num+final_idx]=i
            sg['pred_rel_inds'] = [list(item) for item in np_rel]
            return True
            #del sg['pred_classes'][node_num]

    new_item['pred_boxes'].append(sg['pred_boxes'][node_num])
    new_item['pred_classes'].append(sg['pred_classes'][node_num])

    return False
            #new_sg['nodes'].append(sg['pred_classes'][node_num])
            #new_sg['bboxes'].append(sg['pred_boxes'][node_num])


def merge_function(input_list):
    final_sg_list = []
    base_dict = input_list[0]
    for i, img_idx in enumerate(tqdm(base_dict)):
        new_sg = {}
        new_sg['imgid'] = img_idx
        #new_sg['node_labels'] = []
        for sg_num, sg in enumerate(input_list):
            if sg_num==0:
                new_sg['bboxes'] = sg[img_idx]['pred_boxes']
                new_sg['nodes'] = sg[img_idx]['pred_classes']
                new_sg['edges_label'] = sg[img_idx]['rel_inds']
                new_sg['edges_index'] = sg[img_idx]['pred_rel_inds']
            else:
                final_idx = len(new_sg['nodes'])
                sg[img_idx]['pred_rel_inds'] = [list(item) for item in np.array(sg[img_idx]['pred_rel_inds']) + final_idx]
                new_item = {}
                new_item['pred_boxes'] = []
                new_item['pred_classes'] = []

                for node_num, node in enumerate(sg[img_idx]['pred_classes']):
                    _ = compare_and_add(sg[img_idx], node_num, new_sg, final_idx, new_item)

                new_sg['bboxes'] += new_item['pred_boxes']
                new_sg['nodes'] += new_item['pred_classes']
                new_sg['edges_label'] += sg[img_idx]['rel_inds']
                new_sg['edges_index'] += sg[img_idx]['pred_rel_inds']


        final_sg_list.append(new_sg)
    return final_sg_list


if __name__ == "__main__":
    dicts_butd = pkl.load(open('butd_idx_label_pruned_addrel.pkl', 'rb'))
    dicts_vg200 = pkl.load(open('vg200_idx_label.pkl', 'rb'))
    dicts_vrrvg = pkl.load(open('vrrvg_idx_label.pkl','rb'))
    #vg200_obj_list = json.load(open('vg_obj_list_test.json', 'rb'))

    input_list = [dicts_vrrvg, dicts_vg200]#dicts_vg200]#, dicts_vrrvg]
    output = merge_function(input_list)
    pickle_out = open("vg_vrrvg.pkl", "wb")
    pkl.dump(output, pickle_out)
