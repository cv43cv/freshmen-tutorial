import argparse
import os
from data.datasets.xmlreader import xmlreader
from PIL import Image
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader

class VOC2012Dataset(Dataset):
    def __init__(self, dataset_dir, dataset_xml, dataset_list):
        """
        Args:
            dataset_dir (string) : Path to the img files dir
            dataset_xml (string) : Path to the xml files dir
            dataset_list (string) : Path to the dataset list file

        Return:
            sample (dictionary) : dictionary for a data
                sample['filename'] (string) : filename ~~~.jpg
                sample['image'] () : image matrix
                sample['object'] (dictionary) : dictionary for objects in image
                sample['object'][object class] (dictionary) : dictionary for the object's bounding box coordinates (xmin, ymin, xmax, ymax) 
        """
        self.dataset_dir = dataset_dir
        self.dataset_xml = dataset_xml
        self.dataset_list = dataset_list
        self.data = xmlreader(self.dataset_xml, self.dataset_list)

        self.class_list = ['person', 
            'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.dataset_dir, self.data[idx]['filename'])
        img = cv2.imread(img_name)
        (h, w, _) = img.shape
        sample = self.data[idx].copy()
        for obj_s, box in sample['object'].items():
            if obj_s in self.class_list:
                obj_c = self.class_list.index(obj_s)
                sample['object'][obj_c] = box.copy()

                sample['object'][obj_c]['xmin'] = int(sample['object'][obj_c]['xmin'] * 300 / w)
                sample['object'][obj_c]['xmax'] = int(sample['object'][obj_c]['xmax'] * 300 / w)
                sample['object'][obj_c]['ymin'] = int(sample['object'][obj_c]['ymin'] * 300 / h)
                sample['object'][obj_c]['ymax'] = int(sample['object'][obj_c]['ymax'] * 300 / h)

                sample['object'].pop(obj_s)
        img = cv2.resize(img, (300,300))
        sample['image'] = img

        return sample
                
        
def show_bndbox(sample):
    class_list = ['person', 
            'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']
    img = sample['image']

    for obj, pos in sample['object'].items():
        xmin = pos['xmin']
        ymin = pos['ymin']
        xmax = pos['xmax']
        ymax = pos['ymax']
        img = cv2.putText(img, class_list[obj],(xmin,ymax), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        img = cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,255,0), 2)

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

        print(i, sample['filename'])

        show_bndbox(sample)

        if i==2:
            break

