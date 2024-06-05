import os
from torchvision.datasets.folder import ImageFolder, default_loader

class VTAB(ImageFolder):
    def __init__(self, root, name, train=True, transform=None, target_transform=None, mode=None,is_individual_prompt=False,**kwargs):
        if "caltech_102" in name:
            root = os.path.join(root, "caltech_102")
        elif "cifar_100" in name:
            root = os.path.join(root, "cifar_100")
        elif "dtd_47" in name:
            root = os.path.join(root, "dtd_47")
        elif "flowers_102" in name:
            root = os.path.join(root, "oxford_flowers_102")
        elif "oxford_iiit_pets_37" in name:
            root = os.path.join(root, "oxford_iiit_pets_37")
        elif "clevr_count_8" in name:
            root = os.path.join(root, "clevr_count_8")
        elif "svhn_10" in name:
            root = os.path.join(root, "svhn_10")
        elif "sun_397" in name:
            root = os.path.join(root, "sun_397")
        elif "dmlab_6" in name:
            root = os.path.join(root, "dmlab_6")
        elif "dsprites_ori_16" in name:
            root = os.path.join(root, "dsprites_ori_16")
        elif "patch_camelyon_2" in name:
            root = os.path.join(root, "patch_camelyon_2")
        elif "eurosat_10" in name:
            root = os.path.join(root, "eurosat_10")
        elif "resisc_45" in name:
            root = os.path.join(root, "resisc_45")
        elif "diabetic_retinopathy_5" in name:
            root = os.path.join(root, "diabetic_retinopathy_5")
        elif "clevr_dist_8" in name:
            root = os.path.join(root, "clevr_dist_8")
        elif "kitti_2" in name:
            root = os.path.join(root, "kitti_2")
        elif "dsprites_loc_16" in name:
            root = os.path.join(root, "dsprites_loc_16")
        elif "smallnorb_azi_18" in name:
            root = os.path.join(root, "smallnorb_azi_18")
        elif "smallnorb_ele_18" in name:
            root = os.path.join(root, "smallnorb_ele_18")
        else:
            raise ValueError(f"Unknown dataset name {name}")

        self.dataset_root = root
        self.loader = default_loader
        self.target_transform = None
        self.transform = transform

        
        train_list_path = os.path.join(self.dataset_root, 'train800val200.txt')
        test_list_path = os.path.join(self.dataset_root, 'test.txt')

        
        # train_list_path = os.path.join(self.dataset_root, 'train800.txt')
        # test_list_path = os.path.join(self.dataset_root, 'val200.txt')


        self.samples = []
        if train:
            with open(train_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root,img_name), label))
        else:
            with open(test_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root,img_name), label))