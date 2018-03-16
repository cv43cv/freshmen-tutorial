import argparse
import os
import torch
from data.datasets.xmlreader import xmlreader
from PIL import Image
import numpy as np
import cv2
import random
from torch.utils.data import Dataset, DataLoader

class VOC2012Dataset(Dataset):
    def __init__(self, start, end, dataset_dir, dataset_xml, dataset_list):
        """
        Args:
            dataset_dir (string) : Path to the img files dir
            dataset_xml (string) : Path to the xml files dir
            dataset_list (string) : Path to the dataset list file

        Return:
             
        """
        self.start = start
        self.end = end
        self.num = end - start
        self.dataset_dir = dataset_dir
        self.dataset_xml = dataset_xml
        self.dataset_list = dataset_list
        self.data = xmlreader(self.dataset_xml, self.dataset_list)

        self.class_list = ['background', 
            'person', 
            'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

    def __len__(self):
        return min(self.num, len(self.data))

    def __getitem__(self, idx):
        data = self.data[idx + self.start]
        img_name = os.path.join(self.dataset_dir, data['filename'])
        img = cv2.imread(img_name)
        (h, w, _) = img.shape
        y = [[0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]]
        for i, (obj_s, box) in enumerate(data['object'].items()):
            if i > 2:
                break
            
            obj_c = self.class_list.index(obj_s)

            xmin = box['xmin'] / w
            ymin = box['ymin'] / h
            xmax = box['xmax'] / w
            ymax = box['ymax'] / h
            
            y[i] = [obj_c, xmin, ymin, xmax, ymax]

        y = torch.FloatTensor(y)
        img, y = self.augmentation((img, y))
        
        img = cv2.resize(img, (300,300))
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img.float() / 255           

        sample = (img, y)        

        return sample
    
    def augmentation(self, sample):
        img, y = sample
        p = random.random()

        if p > 0.5:
            img = cv2.flip(img, 1)
            y[:,1:] = torch.mm(y[:,1:], torch.FloatTensor([[0, 0, -1, 0],
                                                            [0, 1, 0, 0],
                                                            [-1, 0, 0, 0],
                                                            [0, 0, 0, 1]]))
            y[:,1:] += torch.FloatTensor([1, 0, 1, 0])
        
        output = (img,y)

        return output

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
    
    voc_dataset = VOC2012Dataset(dataset_dir = dataset_dir, dataset_xml = dataset_xml, dataset_list = dataset_list)

    for i in range(len(voc_dataset)):
        sample = voc_dataset[i]

        show_bndbox(sample)

        if i==2:
            break

