

import argparse
import os
import xml.etree.ElementTree as ET

def xmlreader(dataset_xml, dataset_list):
    f = open(dataset_list,'r')
    data = []
    idx=0
    while True:
        line = f.readline()
        if not line:
            break
        xml = os.path.join(dataset_xml, line[:-1]+'.xml')
        tree = ET.parse(xml)
        root = tree.getroot()
        filename = root.findtext('filename')
        data.append({})
        data[idx]['filename']=filename
        data[idx]['object']={}
        for obj in root.iter('object'):
            if obj.findtext('difficult')==1:
                continue
            objname = obj.findtext('name')
            data[idx]['object'][objname] = {}
            bndbox = obj.find('bndbox')
            for pos in bndbox.iter():
                if pos.tag != 'bndbox':
                    data[idx]['object'][objname][pos.tag] = int(pos.text)
        idx+=1
    f.close()

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
            