import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numbers
import random
import time

from skimage.util.shape import view_as_windows
from skimage.transform import resize

import torchvision

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # The normalize code -> t.sub_(m).div_(s)
        new_tensor = torch.zeros_like(tensor)
        for i, m, s in zip(range(3), self.mean, self.std):
            new_tensor[:, i] = (tensor[:, i] - m) / s
        return new_tensor

# Useful for undoing thetorchvision.transforms.Normalize() 
# From https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # The normalize code -> t.sub_(m).div_(s)
        new_tensor = torch.zeros_like(tensor)
        for i, m, s in zip(range(3), self.mean, self.std):
            new_tensor[:, i] = (tensor[:, i] * s) + m
        return new_tensor

unnorm_fn = UnNormalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

normalize_fn = Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

C = 0

class Noise2Net(object):
    def __init__(self, batch_size, *_args, **_kwargs):
        self.batch_size = batch_size
    
    def do_augmentation(self, imgs):
        global C

        # images: [B, C, H, W]
        net = ResNet(epsilon=0.3).to(device=torch.device('cuda:1'))

        imgs = np.transpose(imgs, (0, 3, 2, 1))
        inputs = torch.from_numpy(imgs).to(device=torch.device('cuda:1')).float()
        inputs = inputs / 255.0

        # print("noise2net input", inputs.shape, torch.max(inputs), torch.min(inputs))
        # torchvision.utils.save_image(inputs[:10].clone().detach(), f"inputs_noise2net_{C}.png")
        
        with torch.no_grad():
            inputs = normalize_fn(inputs)
            outputs = net(inputs)
            outputs = unnorm_fn(outputs).clamp(0, 1)
        
        # torchvision.utils.save_image(outputs[:10].clone().detach(), f"outputs_noise2net_{C}.png")
        # print("noise2net output", outputs.shape)

        # C = C + 1
        # if C > 5:
        #     exit()

        outputs = outputs.data.cpu().numpy() * 255.0
        outputs = np.transpose(outputs, (0, 3, 2, 1))
        return outputs

    def change_randomization_params(self, *_args, **_kwargs):
        pass
    
    def change_randomization_params_all(self, *_args, **_kwargs):
        pass

    def print_params(self):
        print("<Noise2Net params />")

class RandGray(object):
    def __init__(self,  
                 batch_size, 
                 p_rand=0.5,
                 *_args, 
                 **_kwargs):
        
        self.p_gray = p_rand
        self.batch_size = batch_size
        self.random_inds = np.random.choice([True, False], 
                                            batch_size, 
                                            p=[self.p_gray, 1 - self.p_gray])
        
    def grayscale(self, imgs):
        # imgs: b x h x w x c
        b, h, w, c = imgs.shape
        imgs = imgs[:, :, :, 0] * 0.2989 + imgs[:, :, :, 1] * 0.587 + imgs[:, :, :, 2] * 0.114 
        imgs = np.tile(imgs.reshape(b,h,w,-1), (1, 1, 1, 3)).astype(np.uint8)
        return imgs

    def do_augmentation(self, images):
        # images: [B, C, H, W]
        bs, channels, h, w = images.shape    
        if self.random_inds.sum() > 0:
            images[self.random_inds] =  self.grayscale(images[self.random_inds])

        return images
    
    def change_randomization_params(self, index_):
        self.random_inds[index_] = np.random.choice([True, False], 1, 
                                                    p=[self.p_gray, 1 - self.p_gray])
        
    def change_randomization_params_all(self):
        self.random_inds = np.random.choice([True, False], 
                                            self.batch_size, 
                                            p=[self.p_gray, 1 - self.p_gray])
        
    def print_parms(self):
        print(self.random_inds)
        
        
class Cutout(object):
    def __init__(self, 
                 batch_size, 
                 box_min=7, 
                 box_max=22, 
                 pivot_h=12, 
                 pivot_w=24,
                 *_args, 
                 **_kwargs):
        
        self.box_min = box_min
        self.box_max = box_max
        self.pivot_h = pivot_h
        self.pivot_w = pivot_w
        self.batch_size = batch_size
        self.w1 = np.random.randint(self.box_min, self.box_max, batch_size)
        self.h1 = np.random.randint(self.box_min, self.box_max, batch_size)
        
    def do_augmentation(self, imgs):
        n, h, w, c = imgs.shape
        cutouts = np.empty((n, h, w, c), dtype=imgs.dtype)
        for i, (img, w11, h11) in enumerate(zip(imgs, self.w1, self.h1)):
            cut_img = img.copy()
            cut_img[self.pivot_h+h11:self.pivot_h+h11+h11, 
                    self.pivot_w+w11:self.pivot_w+w11+w11, :] = 0
            cutouts[i] = cut_img
        return cutouts
    
    def change_randomization_params(self, index_):
        self.w1[index_] = np.random.randint(self.box_min, self.box_max)
        self.h1[index_] = np.random.randint(self.box_min, self.box_max)

    def change_randomization_params_all(self):
        self.w1 = np.random.randint(self.box_min, self.box_max, self.batch_size)
        self.h1 = np.random.randint(self.box_min, self.box_max, self.batch_size)
        
    def print_parms(self):
        print(self.w1)
        print(self.h1)
        
        
class Cutout_Color(object):
    def __init__(self, 
                 batch_size, 
                 box_min=7, 
                 box_max=22, 
                 pivot_h=12, 
                 pivot_w=24, 
                 obs_dtype='uint8', 
                 *_args, 
                 **_kwargs):
        
        self.box_min = box_min
        self.box_max = box_max
        self.pivot_h = pivot_h
        self.pivot_w = pivot_w
        self.batch_size = batch_size
        self.w1 = np.random.randint(self.box_min, self.box_max, batch_size)
        self.h1 = np.random.randint(self.box_min, self.box_max, batch_size)
        self.rand_box = np.random.randint(0, 255, size=(batch_size, 1, 1, 3), dtype=obs_dtype)
        self.obs_dtype = obs_dtype
        
    def do_augmentation(self, imgs):
        n, h, w, c = imgs.shape
        pivot_h = 12
        pivot_w = 24

        cutouts = np.empty((n, h, w, c), dtype=imgs.dtype)
        for i, (img, w11, h11) in enumerate(zip(imgs, self.w1, self.h1)):
            cut_img = img.copy()
            cut_img[self.pivot_h+h11:self.pivot_h+h11+h11, self.pivot_w+w11:self.pivot_w+w11+w11, :] \
            = np.tile(self.rand_box[i], cut_img[self.pivot_h+h11:self.pivot_h+h11+h11, 
                                                self.pivot_w+w11:self.pivot_w+w11+w11, :].shape[:-1] +(1,))
            cutouts[i] = cut_img
        return cutouts
        
    def change_randomization_params(self, index_):
        self.w1[index_] = np.random.randint(self.box_min, self.box_max)
        self.h1[index_] = np.random.randint(self.box_min, self.box_max)
        self.rand_box[index_] = np.random.randint(0, 255, size=(1, 1, 1, 3), dtype=self.obs_dtype)

    def change_randomization_params_all(self):
        self.w1 = np.random.randint(self.box_min, self.box_max, self.batch_size)
        self.h1 = np.random.randint(self.box_min, self.box_max, self.batch_size)
        self.rand_box = np.random.randint(0, 255, size=(self.batch_size, 1, 1, 3), dtype=self.obs_dtype)
        
    def print_parms(self):
        print(self.w1)
        print(self.h1)
        
class Rand_Flip(object):
    def __init__(self,  
                 batch_size, 
                 p_rand=0.5,
                 *_args, 
                 **_kwargs):
        
        self.p_flip = p_rand
        self.batch_size = batch_size
        self.random_inds = np.random.choice([True, False], 
                                            batch_size, 
                                            p=[self.p_flip, 1 - self.p_flip])
        
    def do_augmentation(self, images):
        if self.random_inds.sum() > 0:
            images[self.random_inds] = np.flip(images[self.random_inds], 2)
        return images
    
    def change_randomization_params(self, index_):
        self.random_inds[index_] = np.random.choice([True, False], 1, 
                                                    p=[self.p_flip, 1 - self.p_flip])

    def change_randomization_params_all(self):
        self.random_inds = np.random.choice([True, False], 
                                            self.batch_size, 
                                            p=[self.p_flip, 1 - self.p_flip])
        
    def print_parms(self):
        print(self.random_inds)
        
class Rand_Rotate(object):
    def __init__(self,  
                 batch_size, 
                 *_args, 
                 **_kwargs):
        
        self.batch_size = batch_size
        self.random_inds = np.random.randint(4, size=batch_size) * batch_size + np.arange(batch_size)
        
    def do_augmentation(self, imgs):
        tot_imgs = imgs
        for k in range(3):
            rot_imgs = np.ascontiguousarray(np.rot90(imgs,k=(k+1),axes=(1,2)))
            tot_imgs = np.concatenate((tot_imgs, rot_imgs), 0)
        return tot_imgs[self.random_inds]
    
    def change_randomization_params(self, index_):
        temp = np.random.randint(4)            
        self.random_inds[index_] = index_ + temp * self.batch_size
        
    def change_randomization_params_all(self):
        self.random_inds = np.random.randint(4, size=self.batch_size) * self.batch_size + np.arange(self.batch_size)
        
    def print_parms(self):
        print(self.random_inds)
        
class Rand_Crop(object):
    def __init__(self,  
                 batch_size, 
                 *_args, 
                 **_kwargs):
        
        self.batch_size = batch_size
        self.crop_size = 64
        self.crop_max = 75 - self.crop_size
        self.w1 = np.random.randint(0, self.crop_max, self.batch_size)
        self.h1 = np.random.randint(0, self.crop_max, self.batch_size)
                
    def do_augmentation(self, imgs):
        # batch size
        n = imgs.shape[0]
        img_size = imgs.shape[1]
        imgs = np.transpose(imgs, (0, 2, 1, 3))
        
        # creates all sliding windows combinations of size (output_size)
        windows = view_as_windows(
            imgs, (1, self.crop_size, self.crop_size, 1))[..., 0,:,:, 0]
        # selects a random window for each batch element
        cropped_imgs = windows[np.arange(n), self.w1, self.h1]
        cropped_imgs = np.swapaxes(cropped_imgs,1,3)
        return cropped_imgs
    
    def change_randomization_params(self, index_):
        self.w1[index_] = np.random.randint(0, self.crop_max)
        self.h1[index_] = np.random.randint(0, self.crop_max)

    def change_randomization_params_all(self):
        self.w1 = np.random.randint(0, self.crop_max, self.batch_size)
        self.h1 = np.random.randint(0, self.crop_max, self.batch_size)
    
    def print_parms(self):
        print(self.w1)
        print(self.h1)
        
class Center_Crop(object):
    def __init__(self, 
                 *_args, 
                 **_kwargs):
        self.crop_size = 64
    
    def do_augmentation(self, image):
        h, w = image.shape[1], image.shape[2]
        new_h, new_w = self.crop_size, self.crop_size

        top = (h - new_h)//2
        left = (w - new_w)//2
        image = image[:, top:top + new_h, left:left + new_w, :]
        return image.copy()
    
    def change_randomization_params(self, index_):
        index_ = index_
        
    def change_randomization_params_all(self):
        index_ = 0
    
    def print_parms(self):
        print('nothing')
        
class ColorJitterLayer(nn.Module):
    def __init__(self, 
                 batch_size,
                 brightness=0.4,                              
                 contrast=0.4,
                 saturation=0.4, 
                 hue=0.5,
                 p_rand=1.0,
                 stack_size=1, 
                 *_args,
                 **_kwargs):
        super(ColorJitterLayer, self).__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        
        self.prob = p_rand
        self.batch_size = batch_size
        self.stack_size = stack_size
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self._device = torch.device('cpu')
    
        # random paramters
        factor_contrast = torch.empty(self.batch_size, device=self._device).uniform_(*self.contrast)
        self.factor_contrast = factor_contrast.reshape(-1,1).repeat(1, self.stack_size).reshape(-1)
        
        factor_hue = torch.empty(self.batch_size, device=self._device).uniform_(*self.hue)
        self.factor_hue = factor_hue.reshape(-1,1).repeat(1, self.stack_size).reshape(-1)
        
        factor_brightness = torch.empty(self.batch_size, device=self._device).uniform_(*self.brightness)
        self.factor_brightness = factor_brightness.reshape(-1,1).repeat(1, self.stack_size).reshape(-1)
        
        factor_saturate = torch.empty(self.batch_size, device=self._device).uniform_(*self.saturation)
        self.factor_saturate = factor_saturate.reshape(-1,1).repeat(1, self.stack_size).reshape(-1)
        

        
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))
        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value
    
    def adjust_contrast(self, x):
        """
            Args:
                x: torch tensor img (rgb type)
            Factor: torch tensor with same length as x
                    0 gives gray solid image, 1 gives original image,
            Returns:
                torch tensor image: Brightness adjusted
        """
        means = torch.mean(x, dim=(2, 3), keepdim=True)
        return torch.clamp((x - means)
                           * self.factor_contrast.view(len(x), 1, 1, 1) + means, 0, 1)
    
    def adjust_hue(self, x):
        h = x[:, 0, :, :]
        h += (self.factor_hue.view(len(x), 1, 1) * 255. / 360.)
        h = (h % 1)
        x[:, 0, :, :] = h
        return x
    
    def adjust_brightness(self, x):
        """
            Args:
                x: torch tensor img (hsv type)
            Factor:
                torch tensor with same length as x
                0 gives black image, 1 gives original image,
                2 gives the brightness factor of 2.
            Returns:
                torch tensor image: Brightness adjusted
        """
        x[:, 2, :, :] = torch.clamp(x[:, 2, :, :]
                                     * self.factor_brightness.view(len(x), 1, 1), 0, 1)
        return torch.clamp(x, 0, 1)
    
    def adjust_saturate(self, x):
        """
            Args:
                x: torch tensor img (hsv type)
            Factor:
                torch tensor with same length as x
                0 gives black image and white, 1 gives original image,
                2 gives the brightness factor of 2.
            Returns:
                torch tensor image: Brightness adjusted
        """
        x[:, 1, :, :] = torch.clamp(x[:, 1, :, :]
                                    * self.factor_saturate.view(len(x), 1, 1), 0, 1)
        return torch.clamp(x, 0, 1)
    
    def transform(self, inputs):
        hsv_transform_list = [rgb2hsv, self.adjust_brightness,
                              self.adjust_hue, self.adjust_saturate,
                              hsv2rgb]
        rgb_transform_list = [self.adjust_contrast]
        
        # Shuffle transform
        if random.uniform(0,1) >= 0.5:
            transform_list = rgb_transform_list + hsv_transform_list
        else:
            transform_list = hsv_transform_list + rgb_transform_list
        for t in transform_list:
            inputs = t(inputs)
        return inputs
    
    def do_augmentation(self, imgs):
        # batch size
        imgs = np.transpose(imgs, (0, 3, 2, 1))
        inputs = torch.from_numpy(imgs).to(self._device).float()
        inputs = inputs / 255.0
        
        outputs = self.forward(inputs)
        outputs = outputs.data.cpu().numpy() * 255.0
        outputs = np.transpose(outputs, (0, 3, 2, 1))
        return outputs
    
    def change_randomization_params(self, index_):
        self.factor_contrast[index_] = torch.empty(1, device=self._device).uniform_(*self.contrast)
        self.factor_hue[index_] = torch.empty(1, device=self._device).uniform_(*self.hue)
        self.factor_brightness[index_] = torch.empty(1, device=self._device).uniform_(*self.brightness)
        self.factor_saturate[index_] = torch.empty(1, device=self._device).uniform_(*self.saturation)

    def change_randomization_params_all(self):
        factor_contrast = torch.empty(self.batch_size, device=self._device).uniform_(*self.contrast)
        self.factor_contrast = factor_contrast.reshape(-1,1).repeat(1, self.stack_size).reshape(-1)
        
        factor_hue = torch.empty(self.batch_size, device=self._device).uniform_(*self.hue)
        self.factor_hue = factor_hue.reshape(-1,1).repeat(1, self.stack_size).reshape(-1)
        
        factor_brightness = torch.empty(self.batch_size, device=self._device).uniform_(*self.brightness)
        self.factor_brightness = factor_brightness.reshape(-1,1).repeat(1, self.stack_size).reshape(-1)
        
        factor_saturate = torch.empty(self.batch_size, device=self._device).uniform_(*self.saturation)
        self.factor_saturate = factor_saturate.reshape(-1,1).repeat(1, self.stack_size).reshape(-1)
        
    def print_parms(self):
        print(self.factor_hue)
        
    def forward(self, inputs):
        # batch size
        random_inds = np.random.choice(
            [True, False], len(inputs), p=[self.prob, 1 - self.prob])
        inds = torch.tensor(random_inds).to(self._device)
        if random_inds.sum() > 0:
            inputs[inds] = self.transform(inputs[inds])
        return inputs

def rgb2hsv(rgb, eps=1e-8):
    # Reference: https://www.rapidtables.com/convert/color/rgb-to-hsv.html
    # Reference: https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py#L287

    _device = rgb.device
    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]

    Cmax = rgb.max(1)[0]
    Cmin = rgb.min(1)[0]
    delta = Cmax - Cmin

    hue = torch.zeros((rgb.shape[0], rgb.shape[2], rgb.shape[3])).to(_device)
    hue[Cmax== r] = (((g - b)/(delta + eps)) % 6)[Cmax == r]
    hue[Cmax == g] = ((b - r)/(delta + eps) + 2)[Cmax == g]
    hue[Cmax == b] = ((r - g)/(delta + eps) + 4)[Cmax == b]
    hue[Cmax == 0] = 0.0
    hue = hue / 6. # making hue range as [0, 1.0)
    hue = hue.unsqueeze(dim=1)

    saturation = (delta) / (Cmax + eps)
    saturation[Cmax == 0.] = 0.
    saturation = saturation.to(_device)
    saturation = saturation.unsqueeze(dim=1)

    value = Cmax
    value = value.to(_device)
    value = value.unsqueeze(dim=1)

    return torch.cat((hue, saturation, value), dim=1)#.type(torch.FloatTensor).to(_device)
    # return hue, saturation, value

def hsv2rgb(hsv):
    # Reference: https://www.rapidtables.com/convert/color/hsv-to-rgb.html
    # Reference: https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py#L287

    _device = hsv.device

    hsv = torch.clamp(hsv, 0, 1)
    hue = hsv[:, 0, :, :] * 360.
    saturation = hsv[:, 1, :, :]
    value = hsv[:, 2, :, :]

    c = value * saturation
    x = - c * (torch.abs((hue / 60.) % 2 - 1) - 1)
    m = (value - c).unsqueeze(dim=1)

    rgb_prime = torch.zeros_like(hsv).to(_device)

    inds = (hue < 60) * (hue >= 0)
    rgb_prime[:, 0, :, :][inds] = c[inds]
    rgb_prime[:, 1, :, :][inds] = x[inds]

    inds = (hue < 120) * (hue >= 60)
    rgb_prime[:, 0, :, :][inds] = x[inds]
    rgb_prime[:, 1, :, :][inds] = c[inds]

    inds = (hue < 180) * (hue >= 120)
    rgb_prime[:, 1, :, :][inds] = c[inds]
    rgb_prime[:, 2, :, :][inds] = x[inds]

    inds = (hue < 240) * (hue >= 180)
    rgb_prime[:, 1, :, :][inds] = x[inds]
    rgb_prime[:, 2, :, :][inds] = c[inds]

    inds = (hue < 300) * (hue >= 240)
    rgb_prime[:, 2, :, :][inds] = c[inds]
    rgb_prime[:, 0, :, :][inds] = x[inds]

    inds = (hue < 360) * (hue >= 300)
    rgb_prime[:, 2, :, :][inds] = x[inds]
    rgb_prime[:, 0, :, :][inds] = c[inds]

    rgb = rgb_prime + torch.cat((m, m, m), dim=1)
    rgb = rgb.to(_device)

    return torch.clamp(rgb, 0, 1)



class GELU(torch.nn.Module):
    def forward(self, x):
        return F.gelu(x)

class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, hidden_planes=24, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = hidden_planes
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale -1
        
        convs = []
        bns = []
        for i in range(self.nums):
            K = random.choice([1, 3, 5])
            D = random.choice([1, 2, 3])
            P = int(((K - 1) / 2) * D)

            convs.append(nn.Conv2d(width, width, kernel_size=K, stride = stride, padding=P, dilation=D, bias=True))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.act = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i==0 or self.stype=='stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]

            sp = self.convs[i](sp)
            sp = self.act(self.bns[i](sp))
          
            if i==0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        
        if self.scale != 1 and self.stype=='normal':
            out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
            out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # out += residual
        # out = self.act(out)

        return out

class ResNet(torch.nn.Module):
    def __init__(self, epsilon=0.2, hidden_planes=24):
        super(ResNet, self).__init__()
        
        self.epsilon = epsilon
        self.hidden_planes = hidden_planes
                
        self.block1 = Bottle2neck(3, 3, hidden_planes=hidden_planes)
        self.block2 = Bottle2neck(3, 3, hidden_planes=hidden_planes)
        self.block3 = Bottle2neck(3, 3, hidden_planes=hidden_planes)
        self.block4 = Bottle2neck(3, 3, hidden_planes=hidden_planes)
        # self.block5 = Bottle2neck(3, 3, hidden_planes=hidden_planes)
        # self.block6 = Bottle2neck(3, 3, hidden_planes=hidden_planes)
    
    def forward_original(self, x):
                
        x = (self.block1(x) * self.epsilon) + x
        x = (self.block2(x) * self.epsilon) + x
        x = (self.block3(x) * self.epsilon) + x
        x = (self.block4(x) * self.epsilon) + x
        
        # if random.random() < 0.5:
        #     x = (self.block5(x) * self.epsilon) + x
        
        # if random.random() < 0.5:
        #     x = (self.block6(x) * self.epsilon) + x
        
        return x

    def forward_randorder(self, x):
        
        num_splits = random.choice([2, 3, 6])
        # print("num_splits = ", num_splits)
        per_split = 6 / num_splits
        # blocks = [self.block1, self.block2, self.block3, self.block4, self.block5, self.block6]
        blocks = [self.block1, self.block2, self.block3, self.block4]
        random.shuffle(blocks)
        
        split_blocks = [blocks[int(round(per_split * i)): int(round(per_split * (i + 1)))] for i in range(num_splits)]
        
        for group in split_blocks:
            group_len = len(group)
            
            branch = x
            for block in group:
                branch = block(branch) * self.epsilon
                branch = branch + ((torch.rand_like(branch) - 0.5) * random.random() * 0.5)
            x = x + branch 
                
        return x
    
    def eval_random_block(self, x):
        # blocks = [self.block1, self.block2, self.block3, self.block4, self.block5, self.block6]
        blocks = [self.block1, self.block2, self.block3, self.block4]
        block = random.choice(blocks)
        return block(x)
    
    def forward_multisplit(self, x):
        
        for section in range(6):
            
            splits = random.choice([1,2,3])
            blocks = random.choice([1,2,3])
            split_output = torch.zeros_like(x)
            
            for split in range(splits):
                branch = x.clone()
                for block in range(blocks):
                    branch = self.eval_random_block(x)
                
                split_output = split_output + (branch * self.epsilon / splits)
            
            x = split_output + x
            
        return x

    def forward(self, x):
        funcs = [
            self.forward_original,
            self.forward_multisplit,
            self.forward_randorder
        ]
        
        random.shuffle(funcs)
        
        F1 = funcs[0]
        F2 = funcs[1]
        F3 = funcs[2]
        return F1(x)
