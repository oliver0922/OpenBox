"""openbox_boxgen: library of the OpenBox pseudo-box generation pipeline
(generate_boxes -> build_pseudo_infos -> track_boxes).

``box_2d_iou`` and ``infos`` load pcdet CUDA extensions that need ``torch``
imported first; both import torch themselves before pcdet.
"""
