import torch
import numpy as np
from loss.common import *

def matched_db(y):
    """
    Args:
        y : (dictionary) ground truth classes for each bounding boxes
    Return :
        array of matched default boxes for each objects (object_no, object_class, layer_depth, i, j, s)
    """

    n_db = [4, 6, 6, 6, 4, 4]
    f_sizes = [38, 19, 10, 5, 3, 1]
    (s_min, s_max) = (0.1, 0.9)
    s = [s_min + (s_max - s_min) * k / 5 for k in range(7)]
    a = [1, 2, 1/2, 3, 1/3]

    box_d = []

    b = 0
    for k, (n, f) in enumerate(zip(n_db, f_sizes)):
        for i in range(f):
            for j in range(f):
                c_y = (i+.5)/f
                c_x = (j+.5)/f
                for r in range(n-1):
                    w = s[k] * a[r]**.5
                    h = s[k] / a[r]**.5
                    box_d.append([c_x-w/2, c_y-h/2, c_x+w/2, c_y+h/2])
                w = (s[k] * s[k+1])**.5 / f
                h = (s[k] * s[k+1])**.5 / f
                box_d.append([c_x-w/2, c_y-h/2, c_x+w/2, c_y+h/2])

    box_d = torch.FloatTensor(box_d)

    obj_list = []
    box_o = []

    for obj_c, box in y['object'].items():
        obj_list.append(obj_c)
        box_o.append([box['xmin']/300, box['ymin']/300, box['xmax']/300, box['ymax']/300])

    box_o = torch.FloatTensor(box_o)

    j = jaccard(box_o, box_d)
    j = j.numpy()
    mbox = np.argwhere(j > 0.5)

    output = []

    for a in range(mbox.shape[0]):
        obj_n = mbox[a,0]
        obj_c = obj_list[obj_n]
        l, i, j, s = ww(mbox[a,1])
        output.append((obj_n, obj_c, l, i ,j, s))

    return output

def ww(x):
    n_db = [4, 6, 6, 6, 4, 4]
    f_sizes = [38, 19, 10, 5, 3, 1]
    
    for l, (n, f) in enumerate(zip(n_db, f_sizes)):
        x -= f**2 * n
        if x < 0:
            x += f**2 * n
            ij =int(x / n)
            s = x - ij * n
            i = int(ij / f)
            j = ij - i * f
            break
    return l, i, j, s



                






def conf_loss(x, y):
    """
    Args:
        x : (tensor) output of conf, Shape (Number of default boxes, C)
        y : (dictionary) ground truth classes for each bounding boxes
    """

    


