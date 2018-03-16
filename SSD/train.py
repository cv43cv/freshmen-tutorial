import os
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import time

from loss import *
from data.dataloader import *
from models.SSD import *


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, help='Dataset img dir')
parser.add_argument('--dataset_xml', type=str, help='Dataset xml dir')
parser.add_argument('--dataset_list', type=str, help='Dataset list file')
args = parser.parse_args()

ssd_dim = 300 
num_classes = 21
num_train=10000
batch_size=32
load = ''
start_iter = 0
max_iter = 10000
learning_rate = 1e-3
weight_decay = 0.0005
gamma = 0.1
momentum = 0.9




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
    dataset = VOC2012Dataset(0, num_train, dataset_dir = args.dataset_dir, dataset_xml = args.dataset_xml, dataset_list = args.dataset_list)
    data_loader = data.DataLoader(dataset, batch_size, shuffle=True, pin_memory=True)
    
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
    criterion = myloss(num_classes = num_classes)

    print('Start training...')
    for iteration in range(start_iter, max_iter):
        ll = 0
        t0 = time.time()
        for i, d in enumerate(data_loader, 0):
            img, y = d

            if torch.cuda.is_available():
                img = Variable(img.cuda())
                y = Variable(y.cuda())
            else:
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
        print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (ll /num_train *batch_size), end=' ')
        print('Timer: %.4f sec.' % (t1 - t0))
        if iteration % 10 == 0:
            torch.save(model.state_dict(), os.path.join('save', 'man-' + str(iteration) + '.pth'))

    print("training end. model is saved in save.pth")

def test():
    dataset = VOC2012Dataset(0, 32, dataset_dir = args.dataset_dir, dataset_xml = args.dataset_xml, dataset_list = args.dataset_list)
    data_loader = data.DataLoader(dataset, batch_size, pin_memory=True)

    model = build_SSD(
                phase = 'train',
                num_classes = num_classes
            )

    model.load_state_dict(torch.load(os.path.join('save',load)))

    if torch.cuda.is_available():
        model = model.cuda()

    for i, d in enumerate(data_loader, 0):
        img, y = d

        if torch.cuda.is_available():
            img = Variable(img.cuda())
            y = Variable(y.cuda())
        else:
            img = Variable(img)
            y = Variable(y)
        
        out = model(img)

        for i in range(img.data.size(0)):
            show_bndbox((img.data[i].cpu(), y.data[i]))
            showtime(img.data[i], (out[0].data[i], out[1].data[i]))


def showtime(img, x):
    """
    Args:
        x : (tuple) x_loc, x_conf
            x_loc : (tensor) shape (num of db, 4)
            x_conf :  (tensor) shape (num of db, C)
    """
    x_loc, x_conf = x

    n_db = [4, 6, 6, 6, 4, 4]
    f_sizes = [38, 19, 10, 5, 3, 1]
    (s_min, s_max) = (0.1, 0.9)
    s = [s_min + (s_max - s_min) * k / 5 for k in range(7)]
    a = [1, 2, 1/2, 3, 1/3]

    C = x_conf.size(1)

    box_d = []
    for k, (n, f) in enumerate(zip(n_db, f_sizes)):
        for i in range(f):
            for j in range(f):
                c_y = (i+.5)/f
                c_x = (j+.5)/f
                for r in range(n-1):
                    w = s[k] * a[r]**.5
                    h = s[k] / a[r]**.5
                    box_d.append([c_x-w/2, c_y-h/2, c_x+w/2, c_y+h/2])
                w = (s[k] * s[k+1])**.5
                h = (s[k] * s[k+1])**.5
                box_d.append([c_x-w/2, c_y-h/2, c_x+w/2, c_y+h/2])

    box_d = torch.FloatTensor(box_d)
    if torch.cuda.is_available():
        box_d = box_d.cuda()

    x_conf = Variable(x_conf)
    soft = F.softmax(x_conf, dim=1).data[:,1:]
    pred, c = torch.max(soft, 1)
    c = c.float().view(-1, 1) + 1
    pred = pred.view(-1, 1)
    x_conf = x_conf.data

    x_loc += minmax_to_cwh(box_d)
    l = cwh_to_minmax(x_loc)

    l = torch.clamp(l, min=1/300, max=1.0)

    box = torch.cat((c, l, pred), 1)

    box = box[torch.sort(box[:,5], descending=True)[1]]
    for i in range(8732):
        if box[i, 5] < 0.1:
            break
        j = jaccard(box[:,1:5],box[i,1:5].view(-1,4))
        idx = (j > 0.2) & (j < 1.0) & (box[:,0]==box[i,0]).view(-1,1)
        box[:, 5][idx] = 0
        box = box[torch.sort(box[:,5], descending=True)[1]]

    idx = box[:,5] > 0.8
    sample = (img.cpu(), box[:,:5][idx.view(-1,1).expand_as(box[:,:5])].contiguous().view(-1,5))
    show_bndbox(sample)


        

    



