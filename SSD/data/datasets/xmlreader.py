

import argparse
import os
import xml.etree.ElementTree as ET

def datareader(dataset_xml, dataset_list):
    
    f = open(dataset_list,'r')
    dataset = []
    
    while True:
        line = f.readline()
        if not line:
            break
        dataset.append(line[:-1])
        
    f.close()

    return dataset

def get_anno(dataset_xml, filename):
    """
    Return:
        sample (dictionary) : dictionary for a data
                sample['filename'] (string) : filename ~~~.jpg
                sample['object'] (dictionary) : dictionary for objects in image
                sample['object'][object class] (dictionary) : dictionary for the object's bounding box coordinates (xmin, ymin, xmax, ymax)
    """
    xml = os.path.join(dataset_xml, filename + '.xml')
    tree = ET.parse(xml)
    root = tree.getroot()
    filename = root.findtext('filename')
    data = {}
    data['filename']=filename
    data['object']=[]
    for i, obj in enumerate(root.iter('object')):
        if obj.findtext('difficult')==1:
            continue
        objname = obj.findtext('name')
        data['object'].append({})
        data['object'][i]['class'] = objname
        bndbox = obj.find('bndbox')
        for pos in bndbox.iter():
            if pos.tag != 'bndbox':
                data['object'][i][pos.tag] = int(pos.text)
    return data



if __name__  == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_xml', type=str, help='Dataset xml dir')
    parser.add_argument('--dataset_list', type=str, help='Dataset list file')
    args = parser.parse_args()
    dataset_xml = args.dataset_xml
    dataset_list = args.dataset_list
    
    data = xmlreader(dataset_xml, dataset_list)
    for i in range(len(data)):
        print(data[i]['filename'])
        for obj, pos in data[i]['object'].items():
            print("  "+obj+"  ", end='')
            print(pos)
            