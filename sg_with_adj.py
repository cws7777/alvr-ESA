import numpy as np
import pickle

sgg_algo = 'scarlett_butd'
dataset = 'butd_vrrvg'
#pwd = '/data/project/rw/SGG/VG_COCO/GRCNN/'
#pwd = '/data/project/rw/CBIR/data/vg_coco/sgg_scarlett/'
#pwd = '/data/project/rw/CBIR/data/vg_coco_vrrvg/'

predictions = pickle.load(open('{}.pkl'.format(dataset), 'rb'))
scene_graphs = []

for i, prediction in enumerate(predictions):
    if i % 1000 == 0:
        print("{}/{}".format(i, len(predictions)))

    num_nodes = 0
    nodes = []
    bboxes = []
    nodes_id = []

    nodes = prediction['nodes']
    num_obj = len(nodes)
    
    assert len(prediction['edges_label']) == len(prediction['edges_index'])
    
    nodes.extend(prediction['edges_label'])
    relations_idx = prediction['edges_index']

    num_nodes = len(nodes)
    adj = np.zeros([num_nodes, num_nodes], dtype=np.uint8)

    for i_rel, rel in enumerate(relations_idx):

        pred_idx = num_obj + i_rel
        sub_idx = rel[0]
        obj_idx = rel[1]
        try:
            adj[sub_idx, pred_idx] = 1
            adj[pred_idx, obj_idx] = 1
        except:
            print(i)

    new_sg = {}

    new_sg['node_labels'] = nodes
    new_sg['adj'] = adj
    #new_sg['filename'] = prediction['filename']
    #new_sg['cocoid'] = prediction['cocoid']
    new_sg['imgid'] = prediction['imgid']
    new_sg['bboxes'] = prediction['bboxes']

    scene_graphs.append(new_sg)

print('data dump to pickle')
pickle.dump(scene_graphs, open('{}_with_adj.pkl'.format(dataset), 'wb'))