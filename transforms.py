import torchvision
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import math
import torch

class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self,img_group):
        return [self.worker(img) for img in img_group] #the normalised hand_label is not chaged for a group scale.


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images

class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)
        self.size = size #just using a crop of size h=w

    def __call__(self, img_group):

        return [self.worker(img) for img in img_group]

class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
        rep_std = self.std * (tensor.size()[0]//len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor

class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L': #doesn't matter if we stack channels first or just do one at a time.
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray): #if we recieve a numpy array stack must have been called i.e using TSM.
            output = torch.from_numpy(pic).permute(2, 0, 1).contiguous() #return a tensor of C*T,H,W
        
        elif type(pic) == list: #we have not called stack and want to return a Tensor of C,T,H,W
            num_imgs = len(pic)
            h,w,c = pic[0].size[1],pic[0].size[0],len(pic[0].mode)
            output = torch.zeros(num_imgs,h,w,c)
            for i,each_pic in enumerate(pic):
            # handle PIL Image
                img = torch.ByteTensor(torch.ByteStorage.from_buffer(each_pic.tobytes()))
                img = img.view(each_pic.size[1], each_pic.size[0], len(each_pic.mode)) #shpe HWC
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
                output[i,:,:,:] = img #shape T,H,W,C
                #img = img.transpose(0, 1).transpose(0, 2).contiguous()
            output = output.permute(3,0,1,2)
        return output.float().div(255) if self.div else img.float()