##############################
#train




#data
DATASET_DIR="/home/j/workspace/freshmen-tutorial/SSD/data/datasets/VOCdevkit/VOC2012/JPEGImages"
DATASET_XML="/home/j/workspace/freshmen-tutorial/SSD/data/datasets/VOCdevkit/VOC2012/Annotations"
DATASET_LIST="/home/j/workspace/freshmen-tutorial/SSD/data/datasets/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt"

python data/dataloader.py \
		   --dataset_dir=$DATASET_DIR \
		   --dataset_xml=$DATASET_XML \
		   --dataset_list=$DATASET_LIST

##############################
