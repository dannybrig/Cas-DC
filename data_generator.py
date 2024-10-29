import numpy as np
import torchvision
from tiny_imgnet import TinyImageNetDataset

def generate_kk_mnist_train_data(data_dir, in_classes):
    
    data_train = torchvision.datasets.MNIST(root=data_dir, train=True, download=False, transform=None)
    train_data = data_train.data
    train_targets = data_train.targets
    
    IN_idxs = np.hstack([np.where(np.array(train_targets) == class_id) for class_id in in_classes]).squeeze()
    IN_data = train_data[IN_idxs]
    IN_targets = np.array(train_targets)[IN_idxs]
    
    map_dict={}
    for new, og in enumerate(np.unique(IN_targets)):
        map_dict[og] = new
    #print(map_dict)
    
    #return IN_data, IN_targets 
    return IN_data, np.array([map_dict[i] for i in IN_targets])

def generate_ku_mnist_train_data(data_dir, out_classes, out_class_id):
    
    OUT_class_id = out_class_id
    
    data_train = torchvision.datasets.MNIST(root=data_dir, train=True, download=False, transform=None)
    train_data = data_train.data
    train_targets = data_train.targets
    
    OUT_idxs = np.hstack([np.where(np.array(train_targets) == class_id) for class_id in out_classes]).squeeze()
    OUT_data = train_data[OUT_idxs]
    OUT_targets = OUT_class_id * np.ones(OUT_idxs.shape[0])
    
    #map_dict={}
    #for new, og in enumerate(np.unique(IN_targets)):
    #    map_dict[og] = new
    #print(map_dict)
    
    return OUT_data, OUT_targets 
    #return IN_data, np.array([map_dict[i] for i in IN_targets])
    
def generate_kk_mnist_test_data(data_dir, in_classes):
    
    data_test = torchvision.datasets.MNIST(root=data_dir, train=False, download=False, transform=None)
    test_data = data_test.data
    test_targets = data_test.targets
    
    IN_idxs = np.hstack([np.where(np.array(test_targets) == class_id) for class_id in in_classes]).squeeze()
    IN_data = test_data[IN_idxs]
    IN_targets = np.array(test_targets)[IN_idxs]
    
    map_dict={}
    for new, og in enumerate(np.unique(IN_targets)):
        map_dict[og] = new
    #print(map_dict)
    
    #return IN_data, IN_targets
    return IN_data, np.array([map_dict[i] for i in IN_targets])

def generate_unknown_mnist_test_data(data_dir, unknown_classes, out_class_id):
    
    OUT_class_id = out_class_id
    
    data_test = torchvision.datasets.MNIST(root=data_dir, train=False, download=False, transform=None)
    test_data = data_test.data
    test_targets = data_test.targets
    
    OUT_idxs = np.hstack([np.where(np.array(test_targets) == class_id) for class_id in unknown_classes]).squeeze()
    OUT_data = test_data[OUT_idxs]
    OUT_targets = OUT_class_id * np.ones(OUT_idxs.shape[0])
    
    return OUT_data, OUT_targets

def generate_kk_svhn_train_data(data_dir, in_classes):
    
    data_train = torchvision.datasets.SVHN(root=data_dir, split='train', download=False, transform=None)
    train_data = data_train.data
    train_targets = data_train.labels
    
    IN_idxs = np.hstack([np.where(np.array(train_targets) == class_id) for class_id in in_classes]).squeeze()
    IN_data = train_data[IN_idxs]
    IN_targets = np.array(train_targets)[IN_idxs]
    
    map_dict={}
    for new, og in enumerate(np.unique(IN_targets)):
        map_dict[og] = new
    #print(map_dict)
    
    #return IN_data, IN_targets 
    return IN_data, np.array([map_dict[i] for i in IN_targets])

def generate_ku_svhn_train_data(data_dir, out_classes, out_class_id):
    
    OUT_class_id = out_class_id
    
    data_train = torchvision.datasets.SVHN(root=data_dir, split='train', download=False, transform=None)
    train_data = data_train.data
    train_targets = data_train.labels
    
    OUT_idxs = np.hstack([np.where(np.array(train_targets) == class_id) for class_id in out_classes]).squeeze()
    OUT_data = train_data[OUT_idxs]
    OUT_targets = OUT_class_id * np.ones(OUT_idxs.shape[0])
    
    #map_dict={}
    #for new, og in enumerate(np.unique(IN_targets)):
    #    map_dict[og] = new
    #print(map_dict)
    
    return OUT_data, OUT_targets 
    #return IN_data, np.array([map_dict[i] for i in IN_targets])
    
def generate_kk_svhn_test_data(data_dir, in_classes):
    
    data_test = torchvision.datasets.SVHN(root=data_dir, split='test', download=False, transform=None)
    test_data = data_test.data
    test_targets = data_test.labels
    
    IN_idxs = np.hstack([np.where(np.array(test_targets) == class_id) for class_id in in_classes]).squeeze()
    IN_data = test_data[IN_idxs]
    IN_targets = np.array(test_targets)[IN_idxs]
    
    map_dict={}
    for new, og in enumerate(np.unique(IN_targets)):
        map_dict[og] = new
    #print(map_dict)
    
    #return IN_data, IN_targets
    return IN_data, np.array([map_dict[i] for i in IN_targets])

def generate_unknown_svhn_test_data(data_dir, unknown_classes, out_class_id):
    
    OUT_class_id = out_class_id
    
    data_test = torchvision.datasets.SVHN(root=data_dir, split='test', download=False, transform=None)
    test_data = data_test.data
    test_targets = data_test.labels
    
    OUT_idxs = np.hstack([np.where(np.array(test_targets) == class_id) for class_id in unknown_classes]).squeeze()
    OUT_data = test_data[OUT_idxs]
    OUT_targets = OUT_class_id * np.ones(OUT_idxs.shape[0])
    
    return OUT_data, OUT_targets

def generate_kk_cifar10_train_data(data_dir, in_classes):
    
    data_train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=None)
    train_data = data_train.data
    train_targets = data_train.targets
    
    IN_idxs = np.hstack([np.where(np.array(train_targets) == class_id) for class_id in in_classes]).squeeze()
    IN_data = train_data[IN_idxs]
    IN_targets = np.array(train_targets)[IN_idxs]
    
    map_dict={}
    for new, og in enumerate(np.unique(IN_targets)):
        map_dict[og] = new
    #print(map_dict)
    
    #return IN_data, IN_targets 
    return IN_data, np.array([map_dict[i] for i in IN_targets])

def generate_ku_cifar10_train_data(data_dir, out_classes, out_class_id):
    
    OUT_class_id = out_class_id
    
    data_train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=None)
    train_data = data_train.data
    train_targets = data_train.targets
    
    OUT_idxs = np.hstack([np.where(np.array(train_targets) == class_id) for class_id in out_classes]).squeeze()
    OUT_data = train_data[OUT_idxs]
    OUT_targets = OUT_class_id * np.ones(OUT_idxs.shape[0])
    
    #map_dict={}
    #for new, og in enumerate(np.unique(IN_targets)):
    #    map_dict[og] = new
    #print(map_dict)
    
    return OUT_data, OUT_targets 
    #return IN_data, np.array([map_dict[i] for i in IN_targets])

def generate_kk_cifar10_test_data(data_dir, in_classes):
    
    data_test = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=None)
    test_data = data_test.data
    test_targets = data_test.targets
    
    IN_idxs = np.hstack([np.where(np.array(test_targets) == class_id) for class_id in in_classes]).squeeze()
    IN_data = test_data[IN_idxs]
    IN_targets = np.array(test_targets)[IN_idxs]
    
    map_dict={}
    for new, og in enumerate(np.unique(IN_targets)):
        map_dict[og] = new
    #print(map_dict)
    
    #return IN_data, IN_targets
    return IN_data, np.array([map_dict[i] for i in IN_targets])

def generate_unknown_cifar10_test_data(data_dir, unknown_classes, out_class_id):
    
    OUT_class_id = out_class_id
    
    data_test = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=None)
    test_data = data_test.data
    test_targets = data_test.targets
    
    OUT_idxs = np.hstack([np.where(np.array(test_targets) == class_id) for class_id in unknown_classes]).squeeze()
    OUT_data = test_data[OUT_idxs]
    OUT_targets = OUT_class_id * np.ones(OUT_idxs.shape[0])
    
    return OUT_data, OUT_targets

def generate_ku_cifar100_train_data(data_dir, out_classes, out_class_id):
    
    OUT_class_id = out_class_id
    
    data_train = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=False, transform=None)
    train_data = data_train.data
    train_targets = data_train.targets
    
    OUT_idxs = np.hstack([np.where(np.array(train_targets) == class_id) for class_id in out_classes]).squeeze()
    OUT_data = train_data[OUT_idxs]
    OUT_targets = OUT_class_id * np.ones(OUT_idxs.shape[0])
    
    #map_dict={}
    #for new, og in enumerate(np.unique(IN_targets)):
    #    map_dict[og] = new
    #print(map_dict)
    
    return OUT_data, OUT_targets 
    #return IN_data, np.array([map_dict[i] for i in IN_targets])

def generate_unknown_cifar100_test_data(data_dir, unknown_classes, out_class_id):
    
    OUT_class_id = out_class_id
    
    data_test = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=False, transform=None)
    test_data = data_test.data
    test_targets = data_test.targets
    
    OUT_idxs = np.hstack([np.where(np.array(test_targets) == class_id) for class_id in unknown_classes]).squeeze()
    OUT_data = test_data[OUT_idxs]
    OUT_targets = OUT_class_id * np.ones(OUT_idxs.shape[0])
    
    return OUT_data, OUT_targets

def generate_kk_tinyimg_train_data(data_dir, in_classes):
    
    data_train = TinyImageNetDataset(root_dir=data_dir, mode='train', preload=True, download=False, load_transform=None)
    train_data = data_train.img_data
    train_targets = data_train.label_data
    
    IN_idxs = np.hstack([np.where(np.array(train_targets) == class_id) for class_id in in_classes]).squeeze()
    IN_data = train_data[IN_idxs]
    IN_targets = np.array(train_targets)[IN_idxs]
    
    map_dict={}
    for new, og in enumerate(np.unique(IN_targets)):
        map_dict[og] = new
    #print(map_dict)
    
    #return IN_data, IN_targets 
    return IN_data, np.array([map_dict[i] for i in IN_targets])

def generate_ku_tinyimg_train_data(data_dir, out_classes, out_class_id):
    
    OUT_class_id = out_class_id
    
    data_train = TinyImageNetDataset(root_dir=data_dir, mode='train', preload=True, download=False, load_transform=None)
    train_data = data_train.img_data
    train_targets = data_train.label_data
    
    OUT_idxs = np.hstack([np.where(np.array(train_targets) == class_id) for class_id in out_classes]).squeeze()
    OUT_data = train_data[OUT_idxs]
    OUT_targets = OUT_class_id * np.ones(OUT_idxs.shape[0])
    
    #map_dict={}
    #for new, og in enumerate(np.unique(IN_targets)):
    #    map_dict[og] = new
    #print(map_dict)
    
    return OUT_data, OUT_targets 
    #return IN_data, np.array([map_dict[i] for i in IN_targets])

def generate_kk_tinyimg_test_data(data_dir, in_classes):
    
    data_test = TinyImageNetDataset(root_dir=data_dir, mode='val', preload=True, download=False, load_transform=None)
    test_data = data_test.img_data
    test_targets = data_test.label_data
    
    IN_idxs = np.hstack([np.where(np.array(test_targets) == class_id) for class_id in in_classes]).squeeze()
    IN_data = test_data[IN_idxs]
    IN_targets = np.array(test_targets)[IN_idxs]
    
    map_dict={}
    for new, og in enumerate(np.unique(IN_targets)):
        map_dict[og] = new
    #print(map_dict)
    
    #return IN_data, IN_targets
    return IN_data, np.array([map_dict[i] for i in IN_targets])

def generate_unknown_tinyimg_test_data(data_dir, unknown_classes, out_class_id):
    
    OUT_class_id = out_class_id
    
    data_test = TinyImageNetDataset(root_dir=data_dir, mode='val', preload=True, download=False, load_transform=None)
    test_data = data_test.img_data
    test_targets = data_test.label_data
    
    OUT_idxs = np.hstack([np.where(np.array(test_targets) == class_id) for class_id in unknown_classes]).squeeze()
    OUT_data = test_data[OUT_idxs]
    OUT_targets = OUT_class_id * np.ones(OUT_idxs.shape[0])
    
    return OUT_data, OUT_targets