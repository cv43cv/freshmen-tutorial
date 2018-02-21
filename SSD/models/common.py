import torch
import numpy

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

if __name__ == '__main__':
    a = numpy.array([[0.,0.,2.,2.],[2.,2.,4.,4.]])
    b = numpy.array([[0.,3.,3.,4.],[1.,1.,4.,3.]])
    
    box_a = torch.from_numpy(a)
    box_b = torch.from_numpy(b)

    print(jaccard(box_a, box_b))