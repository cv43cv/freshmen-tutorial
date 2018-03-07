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
batch_size=32
max_iter = 10000
learning_rate = 1e-4
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
    dataset = VOC2012Dataset(0, 32, dataset_dir = args.dataset_dir, dataset_xml = args.dataset_xml, dataset_list = args.dataset_list)
    data_loader = data.DataLoader(dataset, batch_size, pin_memory=True)
    
    print('Building model...')
    model = build_SSD(
                phase = 'train',
                num_classes = num_classes
            )

    if torch.cuda.is_available():
        model = model.cuda()

    print('initializing model...')
    model.extras.apply(weights_init)
    model.loc.apply(weights_init)
    model.conf.apply(weights_init)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=momentum, weight_decay=weight_decay)
    criterion = myloss(num_classes = num_classes)

    print('Start training...')
    for iteration in range(max_iter):
        for i, d in enumerate(data_loader, 0):

            img, y = d

            if torch.cuda.is_available():
                img = Variable(img.cuda())
                y = Variable(y.cuda())
            else:
                img = Variable(img)
                y = Variable(y)

            t0 = time.time()
            out = model(img)
            optimizer.zero_grad()
            loss_loc, loss_conf = criterion(out, y)
            loss = loss_loc + loss_conf
            if loss.data[0] == 0:
                continue
            loss.backward()
            optimizer.step()
            t1 = time.time()
        if iteration % 100 == 0:
            print('Timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % loss, end=' ')
    
    torch.save(model.state_dict(), 'save.pth')

def test():
    dataset = VOC2012Dataset(32, 40, dataset_dir = args.dataset_dir, dataset_xml = args.dataset_xml, dataset_list = args.dataset_list)
    data_loader = data.DataLoader(dataset, batch_size, pin_memory=True)

    model = build_SSD(
                phase = 'train',
                num_classes = num_classes
            )

    model.load_state_dict(torch.load('save.pth'))

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
            showtime(img.data[i], (out[0].data[i], out[1].data[i]), 0.8)


def showtime(img, x, conf_th):
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
                w = (s[k] * s[k+1])**.5 / f
                h = (s[k] * s[k+1])**.5 / f
                box_d.append([c_x-w/2, c_y-h/2, c_x+w/2, c_y+h/2])

    box_d = torch.FloatTensor(box_d)
    if torch.cuda.is_available():
        box_d = box_d.cuda()

    x_conf = Variable(x_conf)
    box_c = F.softmax(x_conf).data.view(-1, C)[:,1:] > conf_th
    x_conf = x_conf.data

    if box_c.sum() != 0: 
        box_idx = box_c.sum(1).gt(0).unsqueeze(1).expand_as(x_conf)
        _, c_idx = torch.sort(x_conf[box_idx].view(-1,C), descending=True)
        c = c_idx[:,0].contiguous().view(-1,1)
        c = c.float()

        x_loc += minmax_to_cwh(box_d)
        box_idx = box_c.sum(1).gt(0).unsqueeze(1).expand_as(x_loc)
        l = cwh_to_minmax(x_loc[box_idx].view(-1,4))

        box = torch.cat((c,l), 1)

    sample = (img.cpu(), box)
    show_bndbox(sample)



