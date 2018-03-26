import torch

def make_box_d():
    n_db = [4, 6, 6, 6, 4, 4]
    f_sizes = [38, 19, 10, 5, 3, 1]
    (s_min, s_max) = (0.1, 0.9)
    s = [s_min + (s_max - s_min) * k / 5 for k in range(7)]
    a = [1, 2, 1/2, 3, 1/3]

    box_d = []
    for k, (n, f) in enumerate(zip(n_db, f_sizes)):
        for i in range(f):
            for j in range(f):
                c_y = (i+.5)/f
                c_x = (j+.5)/f
                for r in range(n-1):
                    w = s[k] * a[r]**.5
                    h = s[k] / a[r]**.5
                    box_d.append([c_x-w/2, c_y-h/2, c_x+w/2, c_y+h/2])
                w = (s[k] * s[k+1])**.5
                h = (s[k] * s[k+1])**.5
                box_d.append([c_x-w/2, c_y-h/2, c_x+w/2, c_y+h/2])

    box_d = torch.FloatTensor(box_d)
    if torch.cuda.is_available():
        box_d = box_d.cuda()

    return box_d


def make_bb(x_loc):
    x = x_loc.view(-1,4)
    box_d = make_box_d()
    
    l = cwh_to_minmax(x).view_as(x_loc)

    l += box_d

    return l

def minmax_to_cwh(a):
    """
    Args:
        a : (tensor) Shape (n, 4)
    """
    m = [ [.5, 0, -1, 0],
          [0, .5, 0, -1],
          [.5, 0, 1, 0], 
          [0, .5, 0, 1] ]
    m = torch.FloatTensor(m)
    if torch.cuda.is_available():
        m = m.cuda()
    b = torch.mm(a, m)

    return b

def cwh_to_minmax(a):
    """
    Args:
        a : (tensor) Shape (n, 4)
    """
    m = [ [1, 0, 1, 0],
          [0, 1, 0, 1],
          [-.5, 0, .5, 0], 
          [0, -.5, 0, .5] ]
    m = torch.FloatTensor(m)
    if torch.cuda.is_available():
        m = m.cuda()
    b = torch.mm(a, m)

    return b
