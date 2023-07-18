
from typing import Any
from paddle.vision.transforms import ColorJitter,Grayscale
import paddle.vision.transforms as transforms
import random
import math
import cv2
from PIL import Image,ImageFilter
import numpy as np
import paddle
class StrongAugment1(object):
    def __init__(self,**kwargs) -> None:
        augmentation = []

        augmentation.append(RandomColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8))
        augmentation.append(RandomGrayscale(p=0.2))
        augmentation.append(RandomGaussianBlur([0.1,2.0],p=0.5))
       
        # augmentation.append(SaltPepperNoise(0.08,p=1))
       
        # augmentation.append(RandomColorJitter(0.4, 0.4, 0.4, 0.1, p=0.4))
        # augmentation.append(RandomGrayscale(p=0.1))
        # augmentation.append(RandomGaussianBlur([0.1,2.0],p=0.3))
        self.transform = transforms.Compose(augmentation)
        
    def __call__(self, data):
        img = data['image']
        image_pil = Image.fromarray(img.astype("uint8"), "RGB")
        data['strong_image'] = self.transform(image_pil)
        # data['image'] = self.transform(image_pil)
        return data

class StrongAugment2(object):
    def __init__(self,**kwargs) -> None:
        randcrop_transform = transforms.Compose([
            # transforms.ToTensor(),
            RandomErasing(p=0.7,scale=(0.01,0.03),ratio=(0.3,4),value='random'),
            RandomErasing(p=0.5,scale=(0.005,0.02),ratio=(0.1,6),value='random'),
            RandomErasing(p=0.3,scale=(0.005,0.02),ratio=(0.05,8),value='random'),
            # RandomErasing(p=0.3,scale=(0.01,0.03),ratio=(0.3,4),value='random'),
            # RandomErasing(p=0.2,scale=(0.005,0.02),ratio=(0.1,6),value='random'),
            # RandomErasing(p=0.1,scale=(0.005,0.02),ratio=(0.05,8),value='random'),
        ])
        self.transform = randcrop_transform
        
    def __call__(self, data):
        img = data['strong_image']
        data['strong_image'] = self.transform(np.array(img))
        return data


class StrongAugment3(object):
    def __init__(self, **kwargs) -> None:
        augmentation = []
        augmentation.append(RandomColorJitter(0.4, 0.4, 0.4, 0.1, p=0.5))
        augmentation.append(RandomGrayscale(p=0.2))
        augmentation.append(RandomGaussianBlur([0.1,0.2],p=0.3))
        # augmentation.append(RandomColorJitter(0.4, 0.4, 0.4, 0.1, p=0.4))
        # augmentation.append(RandomGrayscale(p=0.1))
        # augmentation.append(RandomGaussianBlur([0.1,2.0],p=0.3))
        self.transform = transforms.Compose(augmentation)
        
    def __call__(self, data):
        img = data['image']
        image_pil = Image.fromarray(img.astype("uint8"), "RGB")
        data['strong_image'] = self.transform(image_pil)
        # data['image'] = self.transform(image_pil)
        return data

class StrongAugment4(object):
    def __init__(self, **kwargs) -> None:
        randcrop_transform = transforms.Compose([
            # transforms.ToTensor(),
            RandomErasing(p=0.3,scale=(0.01,0.02),ratio=(0.3,4),value='random'),
            RandomErasing(p=0.2,scale=(0.005,0.01),ratio=(0.1,6),value='random'),
            RandomErasing(p=0.1,scale=(0.005,0.01),ratio=(0.05,8),value='random'),
            # RandomErasing(p=0.3,scale=(0.01,0.03),ratio=(0.3,4),value='random'),
            # RandomErasing(p=0.2,scale=(0.005,0.02),ratio=(0.1,6),value='random'),
            # RandomErasing(p=0.1,scale=(0.005,0.02),ratio=(0.05,8),value='random'),
        ])
        self.transform = randcrop_transform
        
    def __call__(self, data):
        img = data['strong_image']
        data['strong_image'] = self.transform(img)
        return data
class RandomColorJitter(object):
    def __init__(self,brightness=0, contrast=0, saturation=0, hue=0,p=0):
        self.p = p
        self.aug =ColorJitter(brightness, contrast, saturation, hue)
        
    def __call__(self, data):
        if random.random() < self.p:
            return self.aug(data)
        return data

class RandomGrayscale(object):
    def __init__(self,p=0.2):
        self.p = p
        self.aug = Grayscale(num_output_channels=3)
    def __call__(self, data):
        if random.random() < self.p:
            return self.aug(data)
        return data

class RandomGaussianBlur(object):
    def __init__(self,sigma=[0.1,2.0],p=0.5):
        self.sigma = sigma
        self.p = p
    def __call__(self, data):
        if random.random() > self.p:
            return data
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        # kernel_size_list = [1,3,5,7]
        # index = random.randrange(len(kernel_size_list))
        # kernel_size = (kernel_size_list[index],kernel_size_list[index])
        # img = cv2.GaussianBlur(data,kernel_size,sigma)
        img = data.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img
        
class RandomErasing(object):
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
       
    def __call__(self, img):
        if random.random() > self.p:
            return img
        img_h, img_w,img_c = img.shape
        area = img_h * img_w
        for _ in range(10):
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img_w and h < img_h:
                if self.value == 'random':
                    v = np.random.randn(h,w,img_c)
                x = random.randint(0, img_h - h)
                y = random.randint(0, img_w - w)
                
                img[ x:x + h, y:y + w, :] = v
                # if img.shape == 3:
                #     #img[0, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                #     #img[1, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                #     #img[2, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                #     img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                #     img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                #     img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                #     #img[:, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(3, h, w))
                # else:
                #     img[0, x1:x1+h, y1:y1+w] = self.mean[1]
                #     # img[0, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(1, h, w))
                return img

        return img



class SaltPepperNoise(object):
    def __init__(self,snr,p=0.2) -> None:
        self.snr = snr
        self.p = p
    
    def __call__(self, data) -> Any:
        if random.random() > self.p:
            return data
        img_ = np.array(data).copy()
        h, w, c = img_.shape
        signal_pct = self.snr
        noise_pct = (1 - self.snr)
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
        mask = np.repeat(mask, c, axis=2)
        print((mask==1).sum())
        print((mask==2).sum())
        img_[mask == 1] = 255   
        img_[mask == 2] = 0    
        # return Image.fromarray(img_.astype('uint8')).convert('RGB')
        return img_

# a = StrongAugment1(1,1,1)
# b = StrongAugment2(1,1,1)
# import time
# for i in range(10):
#     img = cv2.imread('/tmp/Experiments/PaddleOCR/train_data/TD_TR/TD500/test_images/IMG_0059.JPG')
#     img1 = {}
#     t = time.time()
#     img1['image'] = img
#     img1 = a(img1)
#     img1 = b(img1)
#     # cv2.imwrite(str(i)+'1232.jpg',img1['image'])
#     print(time.time()-t)