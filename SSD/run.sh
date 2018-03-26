##############################
#train




#data
DATASET1_DIR="/home/j/workspace/freshmen-tutorial/SSD/data/datasets/VOCdevkit/VOC2012/JPEGImages"
DATASET1_XML="/home/j/workspace/freshmen-tutorial/SSD/data/datasets/VOCdevkit/VOC2012/Annotations"
DATASET1_LIST="/home/j/workspace/freshmen-tutorial/SSD/data/datasets/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt"

DATASET2_DIR="/home/j/workspace/freshmen-tutorial/SSD/data/datasets/VOCdevkit/VOC2007/JPEGImages"
DATASET2_XML="/home/j/workspace/freshmen-tutorial/SSD/data/datasets/VOCdevkit/VOC2007/Annotations"
DATASET2_LIST="/home/j/workspace/freshmen-tutorial/SSD/data/datasets/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt"


DATASET3_DIR="/home/j/workspace/freshmen-tutorial/SSD/data/datasets/VOCdevkit/VOC2007_test/JPEGImages"
DATASET3_XML="/home/j/workspace/freshmen-tutorial/SSD/data/datasets/VOCdevkit/VOC2007_test/Annotations"
DATASET3_LIST="/home/j/workspace/freshmen-tutorial/SSD/data/datasets/VOCdevkit/VOC2007_test/ImageSets/Main/test.txt"

python main.py \
		   --dataset1_dir=$DATASET1_DIR \
		   --dataset1_xml=$DATASET1_XML \
		   --dataset1_list=$DATASET1_LIST \
		   --dataset2_dir=$DATASET2_DIR \
		   --dataset2_xml=$DATASET2_XML \
		   --dataset2_list=$DATASET2_LIST \
		   --dataset3_dir=$DATASET3_DIR \
		   --dataset3_xml=$DATASET3_XML \
		   --dataset3_list=$DATASET3_LIST

##############################
