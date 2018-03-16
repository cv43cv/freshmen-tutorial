import torch
from torch.autograd import Variable

def intersect(box_a, box_b):
    """
    Args:
        box_a : (tensor) bounding boxes, Shape (A,4)
        box_b : (tensor) bounding boxes, Shape (B,4)
    Return:
        (tensor) interseting area of bounding boxes, Shape (A,B)
    """

    A = box_a.size(0)
    B = box_b.size(0)

    min_big = torch.min(box_a[:,2:].unsqueeze(1).expand(A,B,2), box_b[:,2:].unsqueeze(0).expand(A,B,2))
    max_small = torch.max(box_a[:,:2].unsqueeze(1).expand(A,B,2), box_b[:,:2].unsqueeze(0).expand(A,B,2))

    inter = torch.clamp((min_big - max_small), min=0)

    return inter[:,:,0] * inter[:,:,1]

def jaccard(box_a, box_b):
    """
    Args:
        box_a : (tensor) bounding boxes, Shape (A,4)
        box_b : (tensor) bounding boxes, Shape (B,4)
    Return:
        (tensor) jaccard overlap of bounding boxes, Shape (A,B)
    """

    A = box_a.size(0)
    B = box_b.size(0)

    area_a = ((box_a[:,2] - box_a[:,0]) * (box_a[:,3] - box_a[:,1])).unsqueeze(1).expand(A,B)
    area_b = ((box_b[:,2] - box_b[:,0]) * (box_b[:,3] - box_b[:,1])).unsqueeze(0).expand(A,B)

    inter = intersect(box_a, box_b)
    union = area_a + area_b - inter

    return inter / union

def match_db(y):
    """
    Args:
        y : tensor shape (N, num_obj, 5)
    Return :
        tensor shape (N, num of db, 5)
    """

    n_db = [4, 6, 6, 6, 4, 4]
    f_sizes = [38, 19, 10, 5, 3, 1]
    (s_min, s_max) = (0.1, 0.9)
    s = [s_min + (s_max - s_min) * k / 5 for k in range(7)]
    a = [1, 2, 1/2, 3, 1/3]

    output = torch.FloatTensor([])
    if torch.cuda.is_available():
        output=output.cuda()

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

    N = y.size(0)

    box_o = y[:,:,1:].contiguous().view(-1,4)
    j = jaccard(box_d, box_o)
    m = torch.eq(torch.max(j,1,keepdim=True)[0],j)
    idx = j > 0.5
    idx = idx * m

    idx = idx.view(-1,y.size(1))
    idx2 = idx.sum(1, keepdim=True).eq(0)
    idx = torch.cat((idx,idx2),1)
    idx = idx.view(j.size(0),-1)
    
    y2 = torch.FloatTensor([0,0,0,0,0])
    if torch.cuda.is_available():
        y2=y2.cuda()
    y2 = y2.unsqueeze(0)
    y2 = y2.unsqueeze(0)
    y2 = y2.expand(y.size(0),1,5)

    y = torch.cat((y,y2),1)

    y = y.view(-1, 5)
    y = torch.unsqueeze(y, 0)
    y = y.expand(j.size(0),-1,5)

    idx = torch.unsqueeze(idx, 2)
    idx = idx.expand_as(y)
    mb = y[idx]

    mb = mb.view(j.size(0),-1,5)
    mb = mb.permute(1,0,2)

    mc = (mb[:,:,1:] - box_d).view(-1,4)
    mc = minmax_to_cwh(mc).view(N,-1,4)
    mb[:,:,1:] = mc
    
    return mb
    """
    for b in range(y.size(0)):
        box_o = y[b, :, 1:]
        j = jaccard(box_d, box_o)
        m = torch.eq(torch.max(j,1,keepdim=True)[0],j)
        idx = j > 0.5
        idx = idx * m
        idx = idx.float()
 
        obj_list = y[b]
        obj_list[:,1:] = minmax_to_cwh(box_o)
    
        mb = torch.mm(idx, obj_list)
        mc = minmax_to_cwh(box_d) * idx.sum(1, keepdim=True).expand_as(box_d)
        mb[:,1:] -= mc
        mb = torch.unsqueeze(mb, 0)

        output = torch.cat((output, mb))
    """

    return output

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

def softmax_loss(x, y):
    """
    Args:
        x : (tensor) shape (N*f, C)
        y : (tensor) shape (N*f,)
    Return:
        softmax loss (tensor) shape (N*f,)
    """
    return log_sum(x) - x.gather(1, y.view(-1,1))

def log_sum(x):
    x_max = x.max(1, keepdim=True)[0]
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max

if __name__ == '__main__':
    a = numpy.array([[0.,0.,2.,2.],[2.,2.,4.,4.]])
    b = numpy.array([[0.,3.,3.,4.],[1.,1.,4.,3.]])
    
    box_a = torch.from_numpy(a)
    box_b = torch.from_numpy(b)

    print(jaccard(box_a, box_b))