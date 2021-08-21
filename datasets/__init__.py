
import contextlib
import os
from collections import namedtuple

from PIL import Image
from torchvision import datasets, transforms

from . import caltech_ucsd_birds
from . import pascal_voc
from .usps import USPS
import torch
import numpy as np
import json

default_dataset_roots = dict(
    MNIST='./data/mnist',
    MNIST_RGB='./data/mnist',
    SVHN='./data/svhn',
    USPS='./data/usps',
    Cifar10='./data/cifar10',
    CUB200='./data/birds',
    PASCAL_VOC='./data/pascal_voc',
    Innov='tub_2026_huanshan',
)

class InnovaDataset(object):
    def __init__(self, root, transforms,phase):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        list_all = list(sorted(os.listdir(os.path.join(root))))
        self.imgs = list(i for i in list_all if ".jpg" in i)
        self.masks = list(i for i in list_all if ".json" in i)
        real_len = len(self.imgs)
        if phase == "train":
            self.imgs = self.imgs[0:int(0.8*real_len)]
            self.masks = self.masks[0:int(0.8*real_len)]
        else:
            self.imgs = self.imgs[int(0.8*real_len):-1]
            self.masks = self.masks[int(0.8*real_len):-1]
    def get_json_record_path(self, ix):
        # return os.path.join(self.path, 'record_'+str(ix).zfill(6)+'.json')  #fill zeros
        return os.path.join(self.root, 'record_' + str(ix) + '.json')  # don't fill zeros
    
    def make_record_paths_absolute(self, record_dict):
        d = {}
        for k, v in record_dict.items():
            if type(v) == str:  # filename
                if '.' in v:
                    v = os.path.join(self.root, v)
            d[k] = v

        return d

    def get_json_record(self, ix):
        path = self.get_json_record_path(ix)
        try:
            with open(path, 'r') as fp:
                json_data = json.load(fp)
        except UnicodeDecodeError:
            raise Exception('bad record: %d. You may want to run `python manage.py check --fix`' % ix)
        except FileNotFoundError:
            raise
        except:
            logger.error('Unexpected error: {}'.format(sys.exc_info()[0]))
            raise

        record_dict = self.make_record_paths_absolute(json_data)
        return record_dict

    def get_record(self, ix):
        json_data = self.get_json_record(ix) # 返回的应该是字典对象
#         data = self.read_record(json_data)
        return json_data

    def read_record(self, record_dict):
        data = {}
        for key, val in record_dict.items():
            typ = self.get_input_type(key)

            # load objects that were saved as separate files
            if typ == 'image_array':
                img = Image.open((val))
                val = np.array(img)

            data[key] = val
        return data

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, self.imgs[idx])
#         mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
#         img = np.array(img,np.float32,copy=False)
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = self.get_record(idx)
        # convert the PIL Image into a numpy array
        label = torch.tensor(mask['user/angle']+1.0, dtype=torch.long)
#         if label == 0: 
#             label =[1,0,0]
#         elif label == 1:
#             label =[0,1,0]
#         else:
#             label =[0,0,1]
# #         img = label
#         label = torch.tensor(label)
        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


dataset_normalization = dict(
    MNIST=((0.1307,), (0.3081,)),
    MNIST_RGB=((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
    USPS=((0.15972736477851868,), (0.25726667046546936,)),
    SVHN=((0.4379104971885681, 0.44398033618927, 0.4729299545288086),
          (0.19803012907505035, 0.2010156363248825, 0.19703614711761475)),
    Cifar10=((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    CUB200=((0.47850531339645386, 0.4992702007293701, 0.4022205173969269),
            (0.23210887610912323, 0.2277066558599472, 0.26652416586875916)),
    PASCAL_VOC=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    Innov=((0.4379104971885681, 0.44398033618927, 0.4729299545288086),
          (0.19803012907505035, 0.2010156363248825, 0.19703614711761475)),
)


dataset_labels = dict(
    MNIST=list(range(10)),
    MNIST_RGB=list(range(10)),
    USPS=list(range(10)),
    SVHN=list(range(10)),
    Cifar10=('plane', 'car', 'bird', 'cat',
             'deer', 'dog', 'monkey', 'horse', 'ship', 'truck'),
    CUB200=caltech_ucsd_birds.class_labels,
    PASCAL_VOC=pascal_voc.object_categories,
    Innov=[0,1,2]
)

# (nc, real_size, num_classes)
DatasetStats = namedtuple('DatasetStats', ' '.join(['nc', 'real_size', 'num_classes']))

dataset_stats = dict(
    MNIST=DatasetStats(1, 28, 10),
    MNIST_RGB=DatasetStats(3, 28, 10),
    USPS=DatasetStats(1, 28, 10),
    SVHN=DatasetStats(3, 32, 10),
    Cifar10=DatasetStats(3, 32, 10),
    CUB200=DatasetStats(3, 224, 200),
    PASCAL_VOC=DatasetStats(3, 224, 20),
    Innov=DatasetStats(3, 224, 3),
)

assert(set(default_dataset_roots.keys()) == set(dataset_normalization.keys()) ==
       set(dataset_labels.keys()) == set(dataset_stats.keys()))


def get_info(state):
    name = state.dataset  # argparse dataset fmt ensures that this is lowercase and doesn't contrain hyphen
    assert name in dataset_stats, 'Unsupported dataset: {}'.format(state.dataset)
    nc, input_size, num_classes = dataset_stats[name]
    normalization = dataset_normalization[name]
    root = state.dataset_root
    if root is None:
        root = default_dataset_roots[name]
    labels = dataset_labels[name]
    return name, root, nc, input_size, num_classes, normalization, labels


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        yield


def get_dataset(state, phase):
    assert phase in ('train', 'test'), 'Unsupported phase: %s' % phase
    name, root, nc, input_size, num_classes, normalization, _ = get_info(state)
    real_size = dataset_stats[name].real_size

    if name == 'MNIST':
        if input_size != real_size:
            transform_list = [transforms.Resize([input_size, input_size], Image.BICUBIC)]
        else:
            transform_list = []
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(*normalization),
        ]
        with suppress_stdout():
            return datasets.MNIST(root, train=(phase == 'train'), download=True,
                                  transform=transforms.Compose(transform_list))
    elif name == 'MNIST_RGB':
        transform_list = [transforms.Grayscale(3)]
        if input_size != real_size:
            transform_list.append(transforms.Resize([input_size, input_size], Image.BICUBIC))
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(*normalization),
        ]
        with suppress_stdout():
            return datasets.MNIST(root, train=(phase == 'train'), download=True,
                                  transform=transforms.Compose(transform_list))
    elif name == 'USPS':
        if input_size != real_size:
            transform_list = [transforms.Resize([input_size, input_size], Image.BICUBIC)]
        else:
            transform_list = []
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(*normalization),
        ]
        with suppress_stdout():
            return USPS(root, train=(phase == 'train'), download=True,
                        transform=transforms.Compose(transform_list))
    elif name == 'SVHN':
        transform_list = []
        if input_size != real_size:
            transform_list.append(transforms.Resize([input_size, input_size], Image.BICUBIC))
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(*normalization),
        ]
        with suppress_stdout():
            return datasets.SVHN(root, split=phase, download=True,
                                 transform=transforms.Compose(transform_list))
    elif name == 'Cifar10':
        transform_list = []
        if input_size != real_size:
            transform_list += [
                transforms.Resize([input_size, input_size], Image.BICUBIC),
            ]
        if phase == 'train':
            transform_list += [
                # TODO: merge the following into the padding options of
                #       RandomCrop when a new torchvision version is released.
                transforms.Pad(padding=4, padding_mode='reflect'),
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip(),
            ]
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(*normalization),
        ]
        with suppress_stdout():
            return datasets.CIFAR10(root, phase == 'train', transforms.Compose(transform_list), download=True)
    elif name == 'CUB200':
        transform_list = []
        if phase == 'train':
            transform_list += [
                transforms.RandomResizedCrop(input_size, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            transform_list += [
                transforms.Resize([input_size, input_size], Image.BICUBIC),
            ]
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(*normalization),
        ]
        return caltech_ucsd_birds.CUB200(root, phase == 'train', transforms.Compose(transform_list), download=True)
    elif name == 'PASCAL_VOC':
        transform_list = []
        if phase == 'train':
            transform_list += [
                transforms.RandomResizedCrop(input_size, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            transform_list += [
                transforms.Resize([input_size, input_size], Image.BICUBIC),
            ]
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(*normalization),
        ]
        if phase == 'train':
            phase = 'trainval'
        return pascal_voc.PASCALVoc2007(root, phase, transforms.Compose(transform_list))

    elif name == "Innov":
        transform_list = []
        transform_list += [
                transforms.Resize([input_size, input_size], Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(*normalization)
            ]
        dataset = InnovaDataset(root, transforms.Compose(transform_list), phase)
        return dataset
    else:
        raise ValueError('Unsupported dataset: %s' % state.dataset)
