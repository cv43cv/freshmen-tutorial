import torch

from loss import *
from data.dataloader import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, help='Dataset img dir')
parser.add_argument('--dataset_xml', type=str, help='Dataset xml dir')
parser.add_argument('--dataset_list', type=str, help='Dataset list file')
args = parser.parse_args()
dataset_dir = args.dataset_dir
dataset_xml = args.dataset_xml
dataset_list = args.dataset_list

voc_dataset = VOC2012Dataset(
    dataset_dir=dataset_dir, dataset_xml=dataset_xml, dataset_list=dataset_list)

print('nnnnn')

i = input()

sample = voc_dataset[int(i)]

print(i, sample['filename'])

show_bndbox(sample)

print(matched_db(sample))
