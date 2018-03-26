import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import time

from loss import *
from utils import *
from data.dataloader import *
from models.SSD import *
from evaluation import *


parser = argparse.ArgumentParser()
parser.add_argument("--dataset1_dir", type=str, help='Dataset1 img dir')
parser.add_argument('--dataset1_xml', type=str, help='Dataset1 xml dir')
parser.add_argument('--dataset1_list', type=str, help='Dataset1 list file')

parser.add_argument("--dataset2_dir", type=str, help='Dataset2 img dir')
parser.add_argument('--dataset2_xml', type=str, help='Dataset2 xml dir')
parser.add_argument('--dataset2_list', type=str, help='Dataset2 list file')

parser.add_argument("--dataset3_dir", type=str, help='Dataset3 img dir')
parser.add_argument('--dataset3_xml', type=str, help='Dataset3 xml dir')
parser.add_argument('--dataset3_list', type=str, help='Dataset3 list file')
args = parser.parse_args()

ssd_dim = 300 
num_classes = 21
num_train1= 32
num_train2= 32
num_train = num_train1 + num_train2
num_test = -1
batch_size=32
load = 'small-290.pth'
start_iter = 0
max_iter = 10000
learning_rate = 1e-3
loss_name = 'CE_loss_hnm'
weight_decay = 0.0005
gamma = 0.1
momentum = 0.9


cudnn.benchmark = True


def xavier(param):
    init.xavier_uniform(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def train():

    loc_loss = 0
    conf_loss = 0
    epoch = 0

    print('Loading Dataset...')
    dataset1 = VOC_Dataset(0, num_train1, dataset_dir = args.dataset1_dir, dataset_xml = args.dataset1_xml, dataset_list = args.dataset1_list, augmentation=False)
    dataset2 = VOC_Dataset(0, num_train2, dataset_dir = args.dataset2_dir, dataset_xml = args.dataset2_xml, dataset_list = args.dataset2_list, augmentation=False)
    data_loader = data.DataLoader(ConcatDataset((dataset2,dataset1)), batch_size, shuffle=True, collate_fn=collate_f)

    dataset3 = VOC_Dataset(0, num_test, dataset_dir = args.dataset3_dir, dataset_xml = args.dataset3_xml, dataset_list = args.dataset3_list, augmentation=False)
    test_loader = data.DataLoader(dataset3, batch_size, shuffle=False, collate_fn=collate_f)

    print('Building model...')
    model = build_SSD(
                phase = 'train',
                num_classes = num_classes
            )

    if torch.cuda.is_available():
        model = model.cuda()

    if load == '':
        print('initializing model...')
        model.extras.apply(weights_init)
        model.loc.apply(weights_init)
        model.conf.apply(weights_init)
    else:    
        print('loading model...')
        model.load_state_dict(torch.load(os.path.join('save',load)))

    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=momentum, weight_decay=weight_decay)
    criterion = my_loss(loss_name, num_classes = num_classes)

    print('Start training...')
    for iteration in range(start_iter, max_iter):
        ll = 0
        t0 = time.time()
        for i, d in enumerate(data_loader, 0):
            _, img, y = d

            img = Variable(img)
            y = Variable(y)
   
            out = model(img)
            optimizer.zero_grad()
            loss_loc, loss_conf = criterion(out, y)

            loss = loss_loc + loss_conf

            if loss.data[0] == 0:
                continue
            ll += loss.data[0]
            loss.backward()
            optimizer.step()

        t1 = time.time()
        print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (ll /len(data_loader)), end=' ')
        print('Timer: %.4f sec.' % (t1 - t0))

        if iteration % 10 == 0:
            torch.save(model.state_dict(), os.path.join('save', 'small-' + str(iteration) + '.pth'))
            l_test = 0
            t0 = time.time()
            """
            for i, d in enumerate(test_loader, 0):
                fi, img, y = d

                img = Variable(img)
                y = Variable(y)
    
                out = model(img)
                optimizer.zero_grad()
                loss_loc, loss_conf = criterion(out, y)
                loss = loss_loc + loss_conf
                if loss.data[0] == 0:
                    continue
                l_test += loss.data[0]
                
            t1 = time.time()
            print('test ' + ' || Loss: %.4f ||' % (l_test /len(test_loader)), end=' ')
            print('Timer: %.4f sec.' % (t1 - t0))
            """

    print("training end. model is saved in save.pth")

def test():
    dataset2 = VOC_Dataset(0, num_train2, dataset_dir = args.dataset2_dir, dataset_xml = args.dataset2_xml, dataset_list = args.dataset2_list, augmentation=False)
    test_loader = data.DataLoader(dataset2, batch_size, shuffle=False, collate_fn=collate_f)

    model = build_SSD(
                phase = 'train',
                num_classes = num_classes
            )

    model.load_state_dict(torch.load(os.path.join('save',load)))

    if torch.cuda.is_available():
        model = model.cuda()

    for c in range(1, num_classes):
        if os.path.exists(os.path.join('result','class'+str(c))):
            os.remove(os.path.join('result','class'+str(c)))

    for i, d in enumerate(test_loader, 0):
        print(i)
        fi, img, y = d

        img = Variable(img)
        y = Variable(y)
        
        out = model(img)

        
        image = img.data
        x = (out[0].data, out[1].data)
        yy = y.data

        #save_result(x, fi, 'class')
    
        for i in range(batch_size):
            showtime(image[i], (make_bb(out[0].data[i]), out[1].data[i]))
    

def voc_eval():
    print('start eval...')
    dataset = VOC_Dataset(0, num_train2, dataset_dir = args.dataset2_dir, dataset_xml = args.dataset2_xml, dataset_list = args.dataset2_list, augmentation=False)
    

    aps = []

    """
    for i in range(1, num_classes):
        aps.append(voc_ap('class',dataset,i)) 

    print(aps)

    aps = np.array(aps)

    mAP = aps.mean()

    print(mAP)
    """
    print(eval_voc_detection('class',dataset))


def voc_ap(savename, dataset, classname, ovthresh=0.5):
    f = open(os.path.join('result',savename+str(classname)), 'r')

    lines = f.readlines()

    f.close()
    
    num_pos = len(lines)
    num_obj = 0
    boxes = {}
    names = []
    preds = []
    tp = np.zeros(num_pos)

    for line in lines:
        data = line.split()
        
        names.append(data[0])
        preds.append(float(data[5]))

    preds = np.array(preds)
    sort = np.argsort(-preds)
    idx = np.argsort(sort)

    for i in range(num_pos):
        data = lines[i].split()[1:5]
        data = [float(x) for x in data]
        if names[i] in boxes:
            boxes[names[i]].append(data+[float(idx[i])])
        else:
            boxes[names[i]] = [data+[float(idx[i])]]

    for k, v in boxes.items():
        bb = torch.FloatTensor(v).view(-1,5)[:,:4]
        if torch.cuda.is_available():
            bb = bb.cuda()
        _, _, y = dataset.feed(k)

        c = y[:,0] == classname
        if c.sum() == 0:
            for i in range(bb.size(0)):
                tp[int(v[i][4])] = 0
            continue
        num_obj += c.sum()
        
        y = y[c.unsqueeze(1).expand_as(y)].view(-1,5)[:,1:]

        j = jaccard(bb, y)

        pos = j >= ovthresh

        for j in range(pos.size(1)):
            for i in range(pos.size(0)):
                if pos[i,j]:
                    tp[int(v[i][4])] = 1
                    break
    fp = 1-tp

    fp = np.cumsum(fp)

    tp = np.cumsum(tp)
    rec = tp / float(num_obj)

    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    
    ap = 0.

    for t in np.arange(0., 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.

    return ap



def save_result(x, fi, savename, max_per_image=200):
    """
    Args:
        x : batch
        fi : batch
    """
    x_loc, x_conf = x

    x_conf = Variable(x_conf)
    soft = F.softmax(x_conf, dim=2).data
    x_conf = x_conf.data

    bb = make_bb(x_loc)

    for c in range(1, num_classes):
        f = open(os.path.join('result',savename+str(c)), 'a')
        for i, fileno in enumerate(fi):
            l = bb[i]
            
            pred = soft[i,:,c]
            idx = pred >= 0.1
            
            if idx.sum() == 0:
                continue
            pred = pred[idx]
            idx = idx.unsqueeze(1).expand_as(l)
            l = l[idx].view(-1,4)

            pred, sort = torch.sort(pred, descending = True)
            l = l[sort]

            l = torch.clamp(l, min=0.0, max=299/300)

            z = l[:,0] < l[:,2]
            if z.sum() == 0:
                continue

            l = l[z.unsqueeze(1).expand_as(l)].view(-1,4)
            pred = pred[z]

            z = l[:,1] < l[:,3]
            if z.sum() == 0:
                continue

            l = l[z.unsqueeze(1).expand_as(l)].view(-1,4)
            pred = pred[z]

            N = l.size(0)

            st = np.zeros((N,N))
            for i in range(N):
                for j in range(N):
                    if i>j:
                        st[i,j] = 1

            st = torch.from_numpy(st).byte()

            if torch.cuda.is_available:
                st = st.cuda()
            
            j = jaccard(l, l)

            idx = j > 0.45
            idx = idx * st
            idx = idx.sum(1).gt(0)
            idx = ~idx

            pred = pred[idx]
            idx = idx.unsqueeze(1).expand_as(l)
            l = l[idx].view(-1,4)
            """
            """
            if l.size(0) > max_per_image:
                l = l[:max_per_image]
                pred = pred[:max_per_image]
                
            for i in range(l.size(0)):
                line = l[i].cpu().numpy()
                line = np.array2string(line, separator = ' ')[1:-1]
                f.write(fileno+' '+line+' '+str(pred[i])+'\n')
            
        f.close()
    
    """
    
    """




def showtime(img, x):
    """
    Args:
        x : (tuple) x_loc, x_conf
            x_loc : (tensor) shape (num of db, 4)
            x_conf :  (tensor) shape (num of db, C)
    """
    x_loc, x_conf = x

    l = make_bb(x_loc)

    x_conf = Variable(x_conf)
    soft = F.softmax(x_conf, dim=1).data[:,1:]
    pred, c = torch.max(soft, 1)
    c = c.float().view(-1, 1) + 1
    pred = pred.view(-1, 1)
    x_conf = x_conf.data

    l = torch.clamp(l, min=0.0, max=299/300)

    box = torch.cat((c, l, pred), 1)

    box = box[torch.sort(box[:,5], descending=True)[1]]

    for i in range(8732):
        if box[i, 5] < 0.1:
            break
        j = jaccard(box[:,1:5],box[i,1:5].view(-1,4))
        idx = (j > 0.2) & (j < 1.0) & (box[:,0]==box[i,0]).view(-1,1)
        box[:, 5][idx] = 0
        box = box[torch.sort(box[:,5], descending=True)[1]]

    idx = box[:,5] > 0.2
    sample = (img.cpu(), box[:,:5][idx.view(-1,1).expand_as(box[:,:5])].contiguous().view(-1,5))
    show_bndbox(sample)


        

    



