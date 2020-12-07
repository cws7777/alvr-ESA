# Toward General Scene Graph: Integration of Visual Semantic Knowlege with Entity Synset Alignment

## Abstract

Scene graph is a graph representation that explicitly represents high-level semantic knowledge of an image such as objects, attributes of objects and relationships between objects. Various tasks have been proposed for the scene graph, but the problem is that they have a limited vocabulary and biased information due to their own hypothesis. Therefore, results of each task are not generalizable and difficult to be applied to other down-stream tasks. In this paper, we propose Entity Synset Alignment(ESA), which is a method to create a general scene graph by aligning various semantic knowledge efficiently to solve this bias problem. 
The ESA uses a large-scale lexical database, WordNet and Intersection of Union (IoU) to align the object labels in multiple scene graphs/semantic knowledge. In experiment, the integrated scene graph is applied to the image-caption retrieval task as a down-stream task. We confirm that integrating multiple scene graphs helps to get better representations of images.

## ACL 2020 Workshop on Advances in Language and Vision Research (https://alvr-workshop.github.io/)

Language and vision research has attracted great attention from both natural language processing (NLP) and computer vision(CV) researcgers. Gradually, this area is shifting from passive perception, templated laguage, and synthetic imagery/environments to active perception, natural language, and photo-realistic simulation or real world deployment. Thus far, few workshops on Language and Vision Research have been organized by groups from the NLP community. We propose the first workshop on Advances in Language and Vision Research (**ALVR**) in order to promote the frontier of language and vision research and to bring interested researchers together to discuss how to best tackle and solve real-world problems in this area.

## Acknowledgement
This work was partly supported by the Institute for Information and Communications Technology Promotion (2015-0-00310-SW.StarLab, 2017-0-01772-VTT, 2018-0-00622-RMI, 2019-0-01367-BabyMind) and Korea Institute for Advancement Technology (P0006720-GENKO) grant funded by the Korea government.

## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.

## Citation
If you use this code, please cite the following paper:
```
@inproceedings{choi-etal-2020-toward,
               title = "Toward General Scene Graph: Integration of Visual Semantic Knowledge with Entity Synset Alignment",
               author = "Choi, Woo Suk  and On, Kyoung-Woon  and  Heo, Yu-Jung  and  Zhang, Byoung-Tak",
               booktitle = "Proceedings of the First Workshop on Advances in Language and Vision Research",
               month = jul,
               year = "2020",
}
```
