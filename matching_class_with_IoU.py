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
    '''
    dicts_save = {}
    dicts_notmatch = {}
    for i, img_idx in enumerate(tqdm(dicts_butd)):
        pred_classes_new = []
        pred_boxes_new = []
        pred_rel_inds_new = []
        rel_inds_new = []
        not_match_pred_class = []
        not_match_pred_boxes = []
        not_match_pred_rel_inds = []
        not_match_rel_inds = []
        
        butd_classes = dicts_butd[img_idx]['pred_classes']
        butd_attribu = dicts_butd[img_idx]['pred_attri']
        butd_boxes = dicts_butd[img_idx]['pred_boxes']
        
        vg200_classes = dicts_vg200[img_idx]['pred_classes']               #'pred_classes': array([ 'chair',  'girl',  
        vg200_pred_rel_inds = dicts_vg200[img_idx]['pred_rel_inds']        #'pred_rel_inds': array([[5, 4], [5, 4],
        vg200_rel_inds = dicts_vg200[img_idx]['rel_inds']             #'rel_inds': array([near, on, next to, 
        vg200_boxes = dicts_vg200[img_idx]['pred_boxes']
        
        class_list = vg200_obj_list[img_idx]['class_lists']
        label_bbox_list = vg200_obj_list[img_idx]['label_bbox']
        numb_index = vg200_obj_list[img_idx]['class_idx']
        
        length_butd_class = len(butd_classes)        
        length_vg200_pred_rel_inds = len(vg200_pred_rel_inds)

        pred_classes_new = class_list.copy()
        pred_boxes_new = label_bbox_list.copy()

        for j in range(length_butd_class):
            butd_object = butd_classes[j]
            butd_match_box = butd_boxes[j]
            butd_match_box[0] = float(butd_match_box[0])
            butd_match_box[1] = float(butd_match_box[1])
            butd_match_box[2] = float(butd_match_box[2])
            butd_match_box[3] = float(butd_match_box[3])
            try:
                
                if (" " in butd_object) == True:
                    tmp_butd_object = butd_object.replace(" ", "_")
                    cb = wordnet.synsets(tmp_butd_object)[0]
                    lemm = cb.lemmas()
                    hypern = cb.hypernyms()
                    hypon = cb.hyponyms()
                    hypernyms_string = [hypern[m].lemma_names() for m in range(len(hypern))]
                    hyponyms_string = [hypon[n].lemma_names() for n in range(len(hypon))]
                    lemm_string = [lemm[o].name() for o in range(len(lemm))]
                else:
                    cb = wordnet.synsets(butd_object)[0]  
                    lemm = cb.lemmas()
                    hypern = cb.hypernyms()
                    hypon = cb.hyponyms()
                    hypernyms_string = [hypern[m].lemma_names() for m in range(len(hypern))]
                    hyponyms_string = [hypon[n].lemma_names() for n in range(len(hypon))]
                    lemm_string = [lemm[o].name() for o in range(len(lemm))]
                    
                if len(hypernyms_string) == 0:
                    #print("here!")
                    hypernyms_string = [['nothing!', 'nothing!'], ['nothing!', 'nothing!']]
                elif len(hyponyms_string) == 0:
                    #print("here!!!!!")
                    hyponyms_string = [['nothing!', 'nothing!'], ['nothing!', 'nothing!']]
            except Exception as ex:
                #print(ex)
                pass
            
            for k in range(len(class_list)):
                #pred_boxes_new.append(label_bbox_list[k])
                #pred_classes_new.append(class_list[k])
                if butd_object == class_list[k]:                           #Checking string first
                    class_bbox = label_bbox_list[k]
                    #pred_boxes_new.append(label_bbox_list[k])
                    #pred_classes_new.append(class_list[k])
                    iou_value = bb_intersection_over_union(class_bbox, butd_match_box)
                    if iou_value > 0.3:                                    #IoU is even same!
                        for l in range(len(vg200_pred_rel_inds)):
                            if numb_index[k] in vg200_pred_rel_inds[l]:
                                pred_rel_inds_new.append(vg200_pred_rel_inds[l])         
                                rel_inds_new.append(vg200_rel_inds[l])

                            
                else:
                    try:
                        if (" " in class_list[k]) == True:
                            tmp_butd_object = class_list[k].replace(" ", "_")
                            cb1 = wordnet.synsets(class_list[k])[0]
                            lemm1 = cb1.lemmas()
                            hypern1 = cb1.hypernyms()
                            hypon1 = cb1.hyponyms()
                            hypernyms_string1 = [hypern1[m].lemma_names() for m in range(len(hypern1))]
                            hyponyms_string1 = [hypon1[n].lemma_names() for n in range(len(hypon1))]
                            lemm_string1 = [lemm1[o].name() for o in range(len(lemm1))]
                        else:
                            cb1 = wordnet.synsets(class_list[k])[0]  
                            lemm1 = cb1.lemmas()
                            hypern1 = cb1.hypernyms()
                            hypon1 = cb1.hyponyms()
                            hypernyms_string1 = [hypern1[m].lemma_names() for m in range(len(hypern1))]
                            hyponyms_string1 = [hypon1[n].lemma_names() for n in range(len(hypon1))]
                            lemm_string1 = [lemm1[o].name() for o in range(len(lemm1))]
                        
                            if len(hyponyms_string1) == 0:
                                hyponym_string1 = [['nothing!', 'nothing!'], ['nothing!', 'nothing!']]
                            elif len(hypernyms_string1) == 0:
                                hypernym_string1 = [['nothing!', 'nothing!'], ['nothing!', 'nothing!']]
                        
                        if (class_list[k] in lemm_string) or (butd_object in lemm_string1):
                            class_bbox = label_bbox_list[k]
                            iou_value = bb_intersection_over_union(class_bbox, butd_match_box)
                            if iou_value > 0.3:                                    #IoU is even same!
                                #pred_classes_new.append(butd_object)
                                #pred_boxes_new.append(butd_match_box)
                                pos_idx = pred_classes_new.index(class_list[k])
                                pred_classes_new[pos_idx] = butd_object
                                for l in range(len(vg200_pred_rel_inds)):
                                    if numb_index[k] in vg200_pred_rel_inds[l]:
                                        pred_rel_inds_new.append(vg200_pred_rel_inds[l])         
                                        rel_inds_new.append(vg200_rel_inds[l])
                        elif (class_list[k] in hypernyms_string[0]) or (butd_object in hypernyms_string1[0]):
                            class_bbox = label_bbox_list[k]
                            iou_value = bb_intersection_over_union(class_bbox, butd_match_box)
                            if iou_value > 0.3:                                    #IoU is even same!
                                pos_idx = pred_classes_new.index(class_list[k])
                                pred_classes_new[pos_idx] = butd_object
                                pred_boxes_new[pos_idx] = butd_match_box

                                for l in range(len(vg200_pred_rel_inds)):
                                    if numb_index[k] in vg200_pred_rel_inds[l]:
                                        pred_rel_inds_new.append(vg200_pred_rel_inds[l])
                                        rel_inds_new.append(vg200_rel_inds[l])

                        elif (class_list[k] in hyponyms_string[0]) or (butd_object in hyponyms_string1[0]):
                            class_bbox = label_bbox_list[k]
                            iou_value = bb_intersection_over_union(class_bbox, butd_match_box)
                            if iou_value > 0.3:                                    #IoU is even same!
                                pos_idx = pred_classes_new.index(class_list[k])
                                pred_classes_new[pos_idx] = butd_object
                                pred_boxes_new[pos_idx] = butd_match_box
                                for l in range(len(vg200_pred_rel_inds)):
                                    if numb_index[k] in vg200_pred_rel_inds[l]:
                                        pred_rel_inds_new.append(vg200_pred_rel_inds[l])
                                        rel_inds_new.append(vg200_rel_inds[l])

                        else:
                            #print("The word is not in the wordnet, sorry!")
                            #not_match_pred_class.append()
                            #not_match_pred_boxes.append(label_bbox_list[k])
                            #not_match_pred_class.append(class_list[k])
                            if butd_object not in pred_classes_new:
                                pred_classes_new.append(butd_object)
                                pred_boxes_new.append(butd_match_box)
                            #for l in range(len(vg200_pred_rel_inds)):
                            #    if numb_index[k] in vg200_pred_rel_inds[l]:
                            #        not_match_pred_rel_inds.append(vg200_pred_rel_inds[l])
                            #        not_match_rel_inds.append(vg200_rel_inds[k])
                    except Exception as ex:
                        pass
            

            
        dicts_save[img_idx] = {'pred_boxes': pred_boxes_new, 'pred_classes': pred_classes_new, 'pred_rel_inds': pred_rel_inds_new, 'rel_inds': rel_inds_new}
        #dicts_notmatch[img_idx] = {'pred_boxes': not_match_pred_boxes, 'pred_classes': not_match_pred_class, 'pred_rel_inds': not_match_pred_rel_inds, 'rel_inds': not_match_rel_inds}

    pickle_out = open("./final/vg-butd.pkl","wb")
    pkl.dump(dicts_save, pickle_out)
    pickle_out.close()    
    #pickle_out1 = open("vg-butd_not.pkl","wb")
    #pkl.dump(dicts_notmatch, pickle_out1)
    #pickle_out1.close()
    '''