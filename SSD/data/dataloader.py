import argparse
import os
import torch
from .datasets.xmlreader import *
from PIL import Image
import numpy as np
import cv2
import random
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET

class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
    
    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        lengths_cumsum = [sum(self.lengths[:i]) for i in range(0, len(self.lengths)+1)]
        for i,cs in enumerate(lengths_cumsum):
            if idx < cs:
                return self.datasets[i-1][idx-lengths_cumsum[i-1]]

class VOC_Dataset(Dataset):
    def __init__(self, start, end, dataset_dir, dataset_xml, dataset_list, augmentation=False):
        """
        Args:
            dataset_dir (string) : Path to the img files dir
            dataset_xml (string) : Path to the xml files dir
            dataset_list (string) : Path to the dataset list file

        Return:
             
        """
        self.start = start
        self.end = end
        self.dataset_dir = dataset_dir
        self.dataset_xml = dataset_xml
        self.dataset_list = dataset_list
        self.dataset = datareader(self.dataset_xml, self.dataset_list)
        self.augmentation = augmentation

        self.class_list = ['background', 
            'person', 
            'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

    def __len__(self):
        if self.end == -1:
            return len(self.dataset)
        elif self.end <= len(self.dataset):
            return self.end-self.start
        else:
            raise ValueError("Dataset end point input is larger than dataset size")

    def __getitem__(self, idx):
        data = self.dataset[idx + self.start]
        return self.feed(data)
        

    def feed(self, filename):
        anno = get_anno(self.dataset_xml, filename)
        img_path = os.path.join(self.dataset_dir, anno['filename'])
        img = cv2.imread(img_path)
        (h, w, _) = img.shape

        y = []
        for i, (obj_s, box) in enumerate(anno['object'].items()):            
            obj_c = self.class_list.index(obj_s)

            xmin = box['xmin'] / w
            ymin = box['ymin'] / h
            xmax = box['xmax'] / w
            ymax = box['ymax'] / h
            
            y.append([obj_c, xmin, ymin, xmax, ymax])

        y = torch.FloatTensor(y)

        if self.augmentation:
            img, y = self.aug((img, y))
        
        img = self.resize(img, (300,300))

        if torch.cuda.is_available():
            img = img.cuda()
            y = y.cuda()           

        sample = (anno['filename'][:-4], img, y)        

        return sample

    def resize(self, img, size):
        img = cv2.resize(img, size)
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img.float() / 255

        return img
    
    def aug(self, sample):
        img, y = sample
        
        img, y = self.crop(img, y, 0.5)
        img, y = self.flip(img, y, 0.5)

        return (img, y)

    def crop(self, img, y, size):
        xmin = size*random.random()
        ymin = size*random.random()
        M = max(xmin, ymin)
        xmax = xmin+1-size+(size-M)*random.random()
        ymax = ymin+1-size+(size-M)*random.random()

        center = torch.mm(y[:,1:], torch.FloatTensor([[0.5, 0],
                                                        [0, 0.5],
                                                        [0.5, 0],
                                                        [0, 0.5]]))
        
        idx = (center[:,0] > xmin) * (center[:,0] < xmax) * (center[:,1] > ymin) * (center[:,1] < ymax)

        if idx.sum() == 0:
            return self.crop(img, y, size)

        y = y[idx.unsqueeze(1).expand_as(y)].view(-1,5)

        y[:,1] = (torch.clamp(y[:,1], min=xmin) - xmin) / (xmax - xmin)
        y[:,2] = (torch.clamp(y[:,2], min=ymin) - ymin) / (ymax - ymin)
        y[:,3] = (torch.clamp(y[:,3], max=xmax) - xmin) / (xmax - xmin)
        y[:,4] = (torch.clamp(y[:,4], max=ymax) - ymin) / (ymax - ymin)

        (h, w, _) = img.shape

        img = img[int(ymin*h):int(ymax*h),int(xmin*w):int(xmax*w),:]

        return (img, y)

    def flip(self, img, y, prob):
        p = random.random()

        if p > prob:
            img = cv2.flip(img, 1)
            y[:,1:] = torch.mm(y[:,1:], torch.FloatTensor([[0, 0, -1, 0],
                                                            [0, 1, 0, 0],
                                                            [-1, 0, 0, 0],
                                                            [0, 0, 0, 1]]))
            y[:,1:] += torch.FloatTensor([1, 0, 1, 0])
        
        return (img,y)

def collate_f(batch):
    names = []
    imgs = []
    boxes = None
    for i, sample in enumerate(batch):
        names.append(sample[0])
        imgs.append(sample[1])
        
        box = sample[2].unsqueeze(0)
        if i == 0:
                boxes = box
        else:
            if box.size(1) > boxes.size(1):
                pad = torch.zeros(boxes.size(0), box.size(1)-boxes.size(1), boxes.size(2))
                if torch.cuda.is_available():
                    pad = pad.cuda()
                boxes = torch.cat((boxes,pad),1)
                boxes = torch.cat((boxes,box),0)
            elif box.size(1) < boxes.size(1):
                pad = torch.zeros(box.size(0), boxes.size(1)-box.size(1), box.size(2))
                if torch.cuda.is_available():
                    pad = pad.cuda()
                box = torch.cat((box,pad),1)
                boxes = torch.cat((boxes,box),0)
            else:
                boxes = torch.cat((boxes,box),0)

    return names, torch.stack(imgs, 0), boxes




def show_bndbox(sample):
    class_list = ['background',
            'person', 
            'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']
    img, box = sample
    img = img.permute(1, 2, 0) * 255
    img = np.uint8(img.numpy())

    color = [(255,0,0), (0,255,0), (0,0,255)]

    
    if box.dim() != 0:
        for i in range(box.size(0)):
            obj = int(box[i, 0])
            if obj == 0:
                continue
            xmin = int(box[i, 1] * 300)
            ymin = int(box[i, 2] * 300)
            xmax = int(box[i, 3] * 300)
            ymax = int(box[i, 4] * 300)
        
            img = cv2.putText(img, class_list[obj],(xmin,ymax), cv2.FONT_HERSHEY_SIMPLEX, 1, color[i%3], 2)
            img = cv2.rectangle(img, (xmin,ymin), (xmax,ymax), color[i%3], 2)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__  == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, help='Dataset img dir')
    parser.add_argument('--dataset_xml', type=str, help='Dataset xml dir')
    parser.add_argument('--dataset_list', type=str, help='Dataset list file')
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    dataset_xml = args.dataset_xml
    dataset_list = args.dataset_list
    
    voc_dataset = VOC2012Dataset(start = 0, end = 32, dataset_dir = dataset_dir, dataset_xml = dataset_xml, dataset_list = dataset_list, augmentation=True)

    for i in range(len(voc_dataset)):
        sample = voc_dataset[i]

        show_bndbox(sample)

