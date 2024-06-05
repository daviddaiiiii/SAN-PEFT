""" Dataset Factory

Hacked together by / Copyright 2021, Ross Wightman
"""
import os
#import hub

from torchvision.datasets import CIFAR100, CIFAR10, MNIST, QMNIST, KMNIST, FashionMNIST, ImageNet, ImageFolder
try:
    from torchvision.datasets import Places365
    has_places365 = True
except ImportError:
    has_places365 = False
try:
    from torchvision.datasets import INaturalist
    has_inaturalist = True
except ImportError:
    has_inaturalist = False

from timm.data.dataset import IterableImageDataset, ImageDataset
import collections

from random import sample as sample_select




# my datasets
from .vtab import VTAB
from .fgvc import FGVC

_TORCH_BASIC_DS = dict(
    cifar10=CIFAR10,
    cifar100=CIFAR100,
    mnist=MNIST,
    qmist=QMNIST,
    kmnist=KMNIST,
    fashion_mnist=FashionMNIST,
)
_TRAIN_SYNONYM = {'train', 'training'}
_EVAL_SYNONYM = {'val', 'valid', 'validation', 'eval', 'evaluation'}

_VTAB_DATASET = [
    "cifar_100", "caltech_102", "dtd_47", "oxford_flowers_102", "oxford_iiit_pets_37", "svhn_10", "sun_397",  # Natural
    "dmlab_6", "dsprites_ori_16", "patch_camelyon_2", "eurosat_10", "resisc_45", "diabetic_retinopathy_5",  # Specialized
    "clevr_count_8", "clevr_dist_8", "dmlab_6", "kitti_2", "dsprites_loc_16", "dsprites_ori_16", "smallnorb_azi_18", "smallnorb_ele_18"  # Structured
]
_FGVC_DATASET = ["stanforddogs_120", "nabirds_555", "cub2011_200", "oxfordflowers_102", "stanfordcars_196"]

def _search_split(root, split):
    # look for sub-folder with name of split in root and use that if it exists
    split_name = split.split('[')[0]
    try_root = os.path.join(root, split_name)
    if os.path.exists(try_root):
        return try_root

    def _try(syn):
        for s in syn:
            try_root = os.path.join(root, s)
            if os.path.exists(try_root):
                return try_root
        return root
    if split_name in _TRAIN_SYNONYM:
        root = _try(_TRAIN_SYNONYM)
    elif split_name in _EVAL_SYNONYM:
        root = _try(_EVAL_SYNONYM)
    return root


def create_dataset(
        name,
        root,
        split='validation',
        search_split=True,
        class_map=None,
        load_bytes=False,
        is_training=False,
        download=False,
        batch_size=None,
        repeats=0,
        few_shot=-1,
        **kwargs
):
    """ Dataset factory method

    In parenthesis after each arg are the type of dataset supported for each arg, one of:
      * folder - default, timm folder (or tar) based ImageDataset
      * torch - torchvision based datasets
      * TFDS - Tensorflow-datasets wrapper in IterabeDataset interface via IterableImageDataset
      * all - any of the above

    Args:
        name: dataset name, empty is okay for folder based datasets
        root: root folder of dataset (all)
        split: dataset split (all)
        search_split: search for split specific child fold from root so one can specify
            `imagenet/` instead of `/imagenet/val`, etc on cmd line / config. (folder, torch/folder)
        class_map: specify class -> index mapping via text file or dict (folder)
        load_bytes: load data, return images as undecoded bytes (folder)
        download: download dataset if not present and supported (TFDS, torch)
        is_training: create dataset in train mode, this is different from the split.
            For Iterable / TDFS it enables shuffle, ignored for other datasets. (TFDS)
        batch_size: batch size hint for (TFDS)
        repeats: dataset repeats per iteration i.e. epoch (TFDS)
        **kwargs: other args to pass to dataset
        
        _VTAB_DATASET = [
    "cifar_100",  # CIFAR-100: 100 classes of tiny images.
    "caltech_102",  # Caltech-102: 102 categories of objects.
    "dtd_47",  # DTD: Describable Textures Dataset with 47 classes.
    "flowers_102",  # Flowers-102: 102 flower categories.
    "pets_37",  # Oxford-IIIT Pets: 37 breeds of pets.
    "clevr_count_8",  # CLEVR Count: Counting objects in CLEVR dataset.
    "svhn_10",  # SVHN: Street View House Numbers with 10 classes.
    "sun_397",  # SUN397: Scene recognition with 397 categories.
    "dmlab_6",  # DMLab: DeepMind Lab with 6 classes.
    "dsprites_ori_16",  # dSprites Orientation: 16 orientations of shapes.
    "patch_camelyon_2",  # PatchCamelyon: Binary classification of histopathology images.
    "eurosat_10",  # EuroSAT: Satellite images with 10 classes.
    "resisc_45",  # RESISC45: Remote sensing image scene classification with 45 classes.
    "diabetic_retinopathy_5",  # Diabetic Retinopathy: 5 stages of diabetic retinopathy.
    "clevr_count_8",  # CLEVR Count: Counting objects in CLEVR dataset.
    "clevr_dist_8",  # CLEVR Distance: Measuring distances between objects in CLEVR.
    "dmlab_6",  # DMLab: DeepMind Lab with 6 classes.
    "kitti_2",  # KITTI: Autonomous driving dataset with 2 classes (car, pedestrian).
    "dsprites_loc_16",  # dSprites Location: 16 locations of shapes.
    "dsprites_ori_16",  # dSprites Orientation: 16 orientations of shapes.
    "smallnorb_azi_18",  # SmallNORB Azimuth: 18 azimuth angles of objects.
    "smallnorb_ele_18"  # SmallNORB Elevation: 18 elevation angles of objects.
    ]

    _FGVC_DATASET = ["stanforddogs_120", # Stanford Dogs: 120 breeds of dogs.
    "nabirds_555",  # NABirds: 555 species of birds.
    "cub2011_200",  # CUB-2011: 200 species of birds.
    "oxfordflowers_102",  # Oxford Flowers: 102 flower categories.
    "stanfordcars_196"  # Stanford Cars: 196 types of cars.
    ]
    
    _TORCH_BASIC_DS = ["cifar10", # CIFAR-10: 10 classes of tiny images.
    "cifar100",  # CIFAR-100: 100 classes of tiny images.
    "mnist",  # MNIST: Handwritten digits with 10 classes.
    "qmist",  # QMNIST: Handwritten digits with 10 classes.
    "kmnist",  # KMNIST: Handwritten characters with 10 classes.
    "fashion_mnist",  # Fashion-MNIST: Fashion products with 10 classes.
    "imagenet",  # ImageNet: 1000 categories of objects.
    "places365",  # Places365: 365 scene categories.
    "inaturalist",  # iNaturalist: 8M images of 8K species.
    "image_folder",  # ImageFolder: Generic image folder dataset.
    "folder"  # ImageFolder: Generic image folder dataset.
    ]

    Returns:
        Dataset object
    """
    name = name.lower()
    if name.startswith('torch/'):
        name = name.split('/', 2)[-1]
        torch_kwargs = dict(root=root, download=download, **kwargs)
        if name in _TORCH_BASIC_DS:
            ds_class = _TORCH_BASIC_DS[name]
            use_train = split in _TRAIN_SYNONYM
            ds = ds_class(train=use_train, **torch_kwargs)
        elif name == 'inaturalist' or name == 'inat':
            assert has_inaturalist, 'Please update to PyTorch 1.10, torchvision 0.11+ for Inaturalist'
            target_type = 'full'
            split_split = split.split('/')
            if len(split_split) > 1:
                target_type = split_split[0].split('_')
                if len(target_type) == 1:
                    target_type = target_type[0]
                split = split_split[-1]
            if split in _TRAIN_SYNONYM:
                split = '2021_train'
            elif split in _EVAL_SYNONYM:
                split = '2021_valid'
            ds = INaturalist(version=split, target_type=target_type, **torch_kwargs)
        elif name == 'places365':
            assert has_places365, 'Please update to a newer PyTorch and torchvision for Places365 dataset.'
            if split in _TRAIN_SYNONYM:
                split = 'train-standard'
            elif split in _EVAL_SYNONYM:
                split = 'val'
            ds = Places365(split=split, **torch_kwargs)
        elif name == 'imagenet':
            if split in _EVAL_SYNONYM:
                split = 'val'
            ds = ImageNet(split=split, **torch_kwargs)
        elif name == 'image_folder' or name == 'folder':
            # in case torchvision ImageFolder is preferred over timm ImageDataset for some reason
            if search_split and os.path.isdir(root):
                # look for split specific sub-folder in root
                root = _search_split(root, split)
            ds = ImageFolder(root, **kwargs)
        else:
            assert False, f"Unknown torchvision dataset {name}"
    elif name.startswith('tfds/'):
        ds = IterableImageDataset(
            root, parser=name, split=split, is_training=is_training,
            download=download, batch_size=batch_size, repeats=repeats, **kwargs)
    else:
        # FIXME support more advance split cfg for ImageFolder/Tar datasets in the future
        # define my datasets
        if name in _VTAB_DATASET:
            ds = VTAB(root=root, name=name, train=is_training, **kwargs)
        elif name in _FGVC_DATASET:
            ds = FGVC(root, name=name, train=is_training, **kwargs)
        # if name == 'stanford_dogs':
        #     ds = dogs(root=root, train=is_training, **kwargs)
        #     # ds = FGVC(root, name="stanford_dogs", train=is_training, **kwargs)
        # elif name == 'nabirds':
        #     ds = NABirds(root=root, train=is_training, **kwargs)
        # elif name == 'cub2011':
        #     ds = Cub2011(root=root, train=is_training, **kwargs)
        # elif name == 'oxford_flowers':
        #     ds = FGVC(root, name="oxford_flowers", train=is_training, **kwargs)
        # elif name == 'stanford_cars':
        #     ds = FGVC(root, name="stanford_cars", train=is_training, **kwargs)
        # elif name in _VTAB_DATASET:
        #     ds = VTAB(root=root, train=is_training, **kwargs)
        else:
            if os.path.isdir(os.path.join(root, split)):
                root = os.path.join(root, split)
            else:
                if search_split and os.path.isdir(root):
                    root = _search_split(root, split)
            if few_shot == -1:
                ds = ImageDataset(root, parser=name, class_map=class_map, load_bytes=load_bytes, **kwargs)
            else:
                ds = ImageDataset_fewshot(root, parser=name, class_map=class_map, load_bytes=load_bytes, few_shot=few_shot, **kwargs)
    return ds


class ImageDataset_fewshot(ImageDataset):
    def __init__(
            self,
            root,
            parser=None,
            class_map=None,
            load_bytes=False,
            few_shot = -1,
            **kwargs
    ):
        super(ImageDataset_fewshot, self).__init__(root, parser=parser, class_map=class_map, load_bytes=load_bytes, **kwargs)

        self.class2samples=collections.OrderedDict()

        for sample in self.parser.samples:
            sample_root, class_num = sample
            if class_num not in self.class2samples:
                self.class2samples[class_num]=[]
            self.class2samples[class_num].append(sample_root)

        samples_fewshot = []
        for class_num, samples in self.class2samples.items():
            selected_samples = sample_select(samples, few_shot)
            samples_fewshot+= list(zip(selected_samples, [class_num]*few_shot))

        self.parser.samples = samples_fewshot

        print(f"++++++++++++++++++++++++++ Selected Sample for Few-Shot Setting: k = {few_shot} ++++++++++++++++++++++++++++")
        print(self.parser.samples)
        print(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
