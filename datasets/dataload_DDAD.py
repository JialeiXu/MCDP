from utils import depth_color
from options_DDAD import DDAD_Options
import PIL.Image as pil
import os,cv2
import random
import numpy as np
from PIL import Image  # using pillow-simd for increased speed
import torch
import torch.utils.data as data
from torchvision import transforms
import json
from torch.utils.data import DataLoader
def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class DDAD_RAWDataset(data.Dataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self,args,num_scales,is_train=False):
        super(DDAD_RAWDataset, self).__init__()
        self.args = args
        self.num_scales = num_scales #4
        self.is_train = is_train
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        if is_train:
            self.filenames = json.load(open(args.json_file))['train']
        else:
            self.filenames = json.load(open(args.json_file_val))['val']
        self.forward_back_l_r_pc = args.train_forward_back_l_r_pc if is_train else args.val_forward_back_l_r_pc
        self.data_path = args.data_path + 'train/' if is_train else args.data_path + 'val_cleaned/'
        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars

        self.resize = transforms.Resize((self.args.height,self.args.width),interpolation=Image.ANTIALIAS)
        self.full_res_shape = (1242, 375)
        self.aug = train_aug if self.is_train else val_aug
        self.mask_img={'CAMERA_01': args.self_mask[0],
                       'CAMERA_05': args.self_mask[1],
                       'CAMERA_06': args.self_mask[2],
                       'CAMERA_07': args.self_mask[3],
                       'CAMERA_08': args.self_mask[4],
                       'CAMERA_09': args.self_mask[5]}

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        #'type, Camera, time, scale'
        json_item = self.filenames[index]
        file_index = np.load(
            self.data_path + str(json_item['timestamp']) + '_' + json_item['Camera'] + '.npz')
        sample = {('color','f',0,0): file_index['rgb'],
                  ('K','f',0 ,-1): file_index['intrinsics'],
                  ('extrinsics','f',0): file_index['extrinsics'],
                  ('depth_gt','f', 0): file_index['depth'],
                  ('pose','f',0,0): file_index['pose']}
        mask_path = './mask/new_mask/'+\
                    str(file_index['Camera'])+'_'+str(json_item['scene'])+'.png'
        sample[('self_mask','f',0)] = cv2.imread(mask_path)[:,:,0]

        if self.forward_back_l_r_pc[0] and self.forward_back_l_r_pc[1]:
            file_index_forward = np.load(
                self.args.data_path + 'train/' + str(json_item['timestamp_forward']) + '_' + json_item[
                    'Camera'] + '.npz')
            sample_forward = {('color','f', 1, 0): file_index_forward['rgb'],
                              ('K','f', 1, -1): file_index_forward['intrinsics'],
                              ('extrinsics','f', 1): file_index_forward['extrinsics'],
                              ('depth_gt','f', 1): file_index_forward['depth'],
                              ('pose','f', 1): file_index_forward['pose']}
            sample.update(sample_forward)

            file_index_back = np.load(
                self.data_path + str(json_item['timestamp_back']) + '_' + json_item['Camera'] + '.npz')
            sample_back =    {('color','f', -1, 0): file_index_back['rgb'],
                              ('K','f', -1, -1): file_index_back['intrinsics'],
                              ('extrinsics','f', -1): file_index_back['extrinsics'],
                              ('depth_gt','f', -1): file_index_back['depth'],
                              ('pose','f', -1): file_index_back['pose']}
            sample.update(sample_back)

        if self.forward_back_l_r_pc[2]:
            file_index_l = np.load(self.data_path + str(json_item['timestamp']) + '_' +
                                   json_item['Camera_l'] + '.npz')
            sample_l = {('color', 'l',0, 0): file_index_l['rgb'],
                              ('K', 'l',0, -1): file_index_l['intrinsics'],
                              ('extrinsics', 'l',0): file_index_l['extrinsics'],
                              ('depth_gt', 'l',0): file_index_l['depth'],
                              ('pose', 'l',0): file_index_l['pose']}
            mask_path = './mask/new_mask/' + \
                        str(file_index['Camera']) + '_' + str(json_item['scene']) + '.png'
            sample_l[('self_mask', 'l', 0)] = cv2.imread(mask_path)[:, :, 0]
            sample.update(sample_l)
        if self.forward_back_l_r_pc[3]:

            file_index_r = np.load(self.data_path + str(json_item['timestamp']) + '_' +
                                   json_item['Camera_r'] + '.npz')
            sample_r = {('color', 'r', 0, 0): file_index_r['rgb'],
                        ('K', 'r', 0, -1): file_index_r['intrinsics'],
                        ('extrinsics', 'r', 0): file_index_r['extrinsics'],
                        ('depth_gt', 'r', 0): file_index_r['depth'],
                        ('pose', 'r', 0): file_index_r['pose']}
            mask_path = './mask/new_mask/' + \
                        str(file_index['Camera']) + '_' + str(json_item['scene']) + '.png'
            sample_r[('self_mask', 'r', 0)] = cv2.imread(mask_path)[:, :, 0]
            sample.update(sample_r)

        sample = self.aug(sample, self.args)
        return sample

def train_aug(sample,args):
    sample_new = {}
    for key,v in sample.items():
        if key[0] == 'K':
            K_4x4 = np.zeros((4, 4))
            K_4x4[:3, :3] = v.copy()[:3, :3]
            K_4x4[3, 3] = 1.
            sample_new[key] = K_4x4
            for scale in range(4):
                scale_K = K_4x4.copy()
                scale_K[0, :] *= 640 / 1936.0 / (2 ** scale)
                scale_K[1, :] *= 384 / 1216.0 / (2 ** scale)
                scale_inv_K = np.linalg.pinv(scale_K)
                sample_new['K',key[1],key[2],scale] = torch.from_numpy(scale_K).float()
                sample_new['inv_K',key[1],key[2],scale] = torch.from_numpy(scale_inv_K).float()
        if key[0]=='extrinsics':
            sample_new['extrinsics'+'_inv',key[1],key[2]] = np.linalg.pinv(v)
        if key[0]=='pose':
            sample_new['pose'+'_inv',key[1],key[2]] = np.linalg.pinv(v)

    do_color_aug = random.random() > 0.5
    if do_color_aug and args.data_aug:
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        color_aug = transforms.ColorJitter(brightness, contrast, saturation, hue)
    else: color_aug = (lambda x:x)

    to_tensor = transforms.ToTensor()
    for key,v in sample.items():
        if key[0] == 'color':
            v = Image.fromarray(v).convert('RGB')
            for scale in range(0,4):
                if scale!=0:
                    s = 2 ** scale
                    resize = transforms.Resize((args.height//s,args.width//s), interpolation=Image.ANTIALIAS)
                else: resize = (lambda x: x)
                v = resize(v)
                sample_new[(key[0], key[1],key[2], scale)] = to_tensor(v)
                sample_new[key[0] + '_aug', key[1],key[2], scale] = to_tensor(color_aug(v))
    for key,v in sample.items():
        if key[0] == 'depth_gt' or key[0]=='self_mask':
            sample_new[key] = to_tensor(v)
    sample.update(sample_new)
    return sample

def val_aug(sample,args=None):
    sample_new = {}
    for key, v in sample.items():
        if key[0] == 'K':
            K_4x4 = np.zeros((4, 4))
            K_4x4[:3, :3] = v.copy()[:3, :3]
            K_4x4[3, 3] = 1.
            sample_new[key] = K_4x4
            for scale in range(4):
                scale_K = K_4x4.copy()
                scale_K[0, :] *= 640 / 1936.0 / (2 ** scale)
                scale_K[1, :] *= 384 / 1216.0 / (2 ** scale)
                scale_inv_K = np.linalg.pinv(scale_K)
                sample_new['K', key[1], key[2],scale] = torch.from_numpy(scale_K).float()
                sample_new['inv_K', key[1], key[2],scale] = torch.from_numpy(scale_inv_K).float()

    to_tensor = transforms.ToTensor()
    for key, v in sample.items():
        if key[0] == 'color':
            v = Image.fromarray(v).convert('RGB')
            for scale in range(0, 4):
                if scale != 0:
                    s = 2 ** scale
                    resize = transforms.Resize((args.height // s, args.width // s), interpolation=Image.ANTIALIAS)
                else:
                    resize = (lambda x: x)
                v = resize(v)
                sample_new[(key[0], key[1], key[2],scale)] = to_tensor(v)
    for key, v in sample.items():
        if key[0] == 'depth_gt' or key[0] == 'self_mask':
            sample_new[key] = to_tensor(v)
        if key[0]=='extrinsics':
            sample_new['extrinsics'+'_inv',key[1],key[2]] = np.linalg.pinv(v)
        if key[0]=='pose':
            sample_new['pose'+'_inv',key[1],key[2]] = np.linalg.pinv(v)

    sample.update(sample_new)
    return sample


if __name__=='__main__':
    options = MonodepthOptions()
    opts = options.parse()
    train_dataset = DDAD_RAWDataset(opts, 4, is_train=True)
    val_dataset = DDAD_RAWDataset(opts, 4, is_train=False)
    item = train_dataset.__getitem__(0)
    train_loader = DataLoader(
        val_dataset, 2)
