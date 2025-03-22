import os
import torch.utils.data as data
import PIL.Image as Image
import pandas as pd
from torchvision import transforms
import torch
import numpy as np

def label2onehot(labels, classes):
    labels_onehot = torch.zeros(labels.size()[0],classes)
    index = labels.view(-1,1)
    labels_onehot.scatter_(dim=1, index = index, value = 1)
    return labels_onehot

def label2onehot_dataloader(dataloader, classes):
    for batch_cnt, (inputs, labels, item) in enumerate(dataloader['train']):
        if batch_cnt == 0:
            train_labels = labels
        else:
            train_labels = torch.cat((train_labels, labels))
    train_tr = label2onehot(train_labels.cpu(), classes).numpy()
    dataloader['train'].dataset.labels = train_tr

    for batch_cnt, (inputs, labels, item) in enumerate(dataloader['val']):
        if batch_cnt == 0:
            val_labels = labels
        else:
            val_labels = torch.cat((val_labels, labels))
    val_tr = label2onehot(val_labels.cpu(), classes).numpy()
    dataloader['val'].dataset.labels = val_tr

    for batch_cnt, (inputs, labels, item) in enumerate(dataloader['base']):
        if batch_cnt == 0:
            base_labels = labels
        else:
            base_labels = torch.cat((base_labels, labels))
    base_tr = label2onehot(base_labels.cpu(), classes).numpy()
    dataloader['base'].dataset.labels = base_tr

def calc_train_codes(dataloader, bits, classes):
    for batch_cnt, (inputs, labels, item) in enumerate(dataloader['base']):
        if batch_cnt == 0:
            train_labels = labels
        else:
            train_labels = torch.cat((train_labels, labels))
    L_tr = label2onehot(train_labels.cpu(), classes).numpy()

    train_size = L_tr.shape[0]
    sigma = 1
    delta = 0.0001
    myiter = 15

    V = np.random.randn(bits, train_size)
    B = np.sign(np.random.randn(bits, train_size))
    S1, E, S2 = np.linalg.svd(np.dot(B, V.T))
    R = np.dot(S1, S2)
    L = L_tr.T

    for it in range(myiter):

        B = -1 * np.ones((bits, train_size))
        B[(np.dot(R, V)) >= 0] = 1

        Ul = np.dot(sigma * np.dot(L, V.T), np.linalg.pinv(sigma * np.dot(V, V.T)))

        V = np.dot(np.linalg.pinv(sigma * np.dot(Ul.T, Ul) + delta * np.dot(R.T, R)),
                   sigma * np.dot(Ul.T, L) + delta * np.dot(R.T, B))

        S1, E, S2 = np.linalg.svd(np.dot(B, V.T))
        R = np.dot(S1, S2)

    B1 = B.T
    B1 = np.sign(B1)
    print('Code generated,', 'size:', B1.shape)
    return B1


class dataset(data.Dataset):
    def __init__(self, set_name, root_dir=None, transforms=None, train=False, test=False):
        self.root_path = root_dir
        self.train = train
        self.test = test
        self.transforms=transforms
        if self.train:
            self.train_anno = pd.read_csv(os.path.join(self.root_path, set_name+'_train.txt'), \
                                      sep=" ", \
                                      header=None, \
                                     names=['ImageName', 'label'])
            self.paths= self.train_anno['ImageName'].tolist()
            #print(self.paths)
            self.labels = self.train_anno['label'].tolist()
        if self.test:
            self.test_anno = pd.read_csv(os.path.join(self.root_path, set_name+'_test.txt'), \
                                      sep=" ", \
                                      header=None, \
                                     names=['ImageName', 'label'])
            self.paths= self.test_anno['ImageName'].tolist()

            self.labels = self.test_anno['label'].tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = os.path.join(self.root_path, self.paths[item])
        img = self.pil_loader(img_path)
        if self.test:
            img = self.transforms(img)
            label = self.labels[item]
            return img, label, item
        if self.train:
            img = self.transforms(img)
            label = self.labels[item]
            return img, label, item
    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

def get_data_fine(config):
    if config["dataset"] == 'cub_bird':
        config["n_class"] = 200
        config["topK"] = -1
        data_dir = '/home/lhc/data/fine_grained_image_retrieval/cub_bird/'
    elif config["dataset"] == 'aircraft':
        config["topK"] = -1
        config["n_class"] = 100
        # data_dir = './datasets/aircraft/'
        data_dir = '/home/lhc/data/fine_grained_image_retrieval/aircraft/'
    elif config["dataset"] == 'Stanford_Cars':
        config["topK"] = -1
        config["n_class"] = 196
        data_dir = '/home/lhc/data/fine_grained_image_retrieval/Stanford_Cars/'
    else:
        print('undefined dataset ! ')
    print('Dataset:', config["dataset"], ', num of classes:', config["n_class"])
    print(data_dir)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomCrop((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_set = dataset(config["dataset"], root_dir=data_dir, transforms=data_transforms['train'], train=True)
    print('train_set', len(train_set))
    val_set = dataset(config["dataset"], root_dir=data_dir, transforms=data_transforms['val'], test=True)
    print('val_set', len(val_set))
    base_set = dataset(config["dataset"], root_dir=data_dir, transforms=data_transforms['val'], train=True)
    print('basa_set', len(base_set))

    dataloader = {}
    dataloader['train'] = data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=8,
                                          pin_memory=True)
    setattr(dataloader['train'], 'total_item_len', len(train_set))
    dataloader['val'] = data.DataLoader(val_set, batch_size=config["batch_size"], shuffle=False, num_workers=8,
                                        pin_memory=True)
    setattr(dataloader['val'], 'total_item_len', len(val_set))
    dataloader['base'] = data.DataLoader(base_set, batch_size=config["batch_size"], shuffle=False, num_workers=8,
                                         pin_memory=True)
    setattr(dataloader['base'], 'total_item_len', len(base_set))
    if config["info"] == '[FISH]' or config["info"] == '[MBLNet]'or config["info"] == '[MBLNet_base]'or config["info"] == '[MBLNet_FFL]'or config["info"] == '[MBLNet_FFL_DSSE]'or config["info"] == '[MBLNet_FFL_DFOL]'or config["info"] == '[MBLNet_hash]'or config["info"] == '[MBLNet_hash_cls]'or config["info"] == '[MBLNet_hash_cls_sig]'or config["info"] == '[MBLNet_hash_cls_sig_obj]':#直接返回原标签数据集不做ont-hot处理
        return dataloader, len(base_set)

    label2onehot_dataloader(dataloader, config['n_class']) #one-hot

    return dataloader["train"], dataloader["val"], dataloader["base"], len(train_set), len(val_set), len(base_set)