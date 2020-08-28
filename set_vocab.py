import pickle
import time
import numpy as np

vocab2idx = {"<UNK>":0}
vocab_objects = []
vocab_relations = []
vocab_attributes = []

idx2vocab = ["<UNK>"]
init_len = len(idx2vocab)
vocab_len = init_len
not_in = 0

print("load: sg")
tic = time.time()
sgg_algo = 'scarlett_butd'
dataset = 'butd_vg_vrrvg'
#pwd = '/data/project/rw/CBIR/data/vg_coco/sgg_scarlett/'
sgs = pickle.load(open('{}_with_adj.pkl'.format(dataset), 'rb'))
print("all sg data is loaded, {}s".format(time.time()-tic))

print("load: glove")
glove_path = '/data/project/rw/VisualGenome/glove_data/glove.6B.300d.pkl'
glove = pickle.load(open(glove_path, 'rb'))
print("loaded: glove")

for i, sg in enumerate(sgs):
    # if i > 1000:
    #     break
    if i % 1000 == 0:
        print("{}/{}".format(i, len(sgs)))

    nodes = sg['node_labels']
    for node in nodes:
        name = node
        if name not in vocab2idx:
            vocab2idx[name] = vocab_len
            idx2vocab.append(name)
            vocab_len += 1

print("total vocab number: {}".format(vocab_len))

glove_embs = np.zeros([vocab_len, 300])
for idx, strs in enumerate(idx2vocab[init_len:]):
    tokens = strs.strip().split(' ')
    if len(tokens) > 1:
        glove_for_phrase = []
        for token in tokens:
            if token in glove:
                glove_for_phrase.append(glove[token])

        if len(glove_for_phrase) > 0:
            glove_for_phrase = np.vstack(glove_for_phrase)
            glove_embs[idx + init_len] = np.mean(glove_for_phrase, axis=0, keepdims=True)

        else:
            print("no glove token for {}".format(tokens))

    else:
        if strs in glove:
            glove_embs[idx+init_len] = glove[strs]
        else:
            #glove_embs[idx+init_len] = np.random.random([300])
            if not_in < 100 and not_in % 10 == 0:
                print(strs + " Not in glove")
            not_in += 1
print("not in words", not_in)

pickle.dump(glove_embs, open(pwd+'glove_embs_{}_sgg_{}.pkl'.format(dataset, sgg_algo), 'wb'))
pickle.dump(vocab2idx, open(pwd+'vocab2idx_{}_sgg_{}.pkl'.format(dataset, sgg_algo), 'wb'))
pickle.dump(idx2vocab, open(pwd+'idx2vocab_{}_sgg_{}.pkl'.format(dataset, sgg_algo), 'wb'))
"""
save_pickle(vocab_objects, project_pwd+'vocab_objects.pkl')
save_pickle(vocab_attributes, project_pwd+'vocab_attributes.pkl')
save_pickle(vocab_relations, project_pwd+'vocab_relations.pkl')
"""