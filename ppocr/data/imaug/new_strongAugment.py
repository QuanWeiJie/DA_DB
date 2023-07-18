
from typing import Any
from paddle.vision.transforms import ColorJitter,Grayscale
import paddle.vision.transforms as transforms
import random
import math
import cv2
from PIL import Image,ImageFilter
import numpy as np
import paddle
class StrongAugment_inplace(object):
    def __init__(self, **kwargs) -> None:
        augmentation1 = []
        augmentation2 = []
        
        augmentation1.append(Random_gaussian_noise(mean=0.1,sigma=[0.1,0.15],p=0.2))
        augmentation1.append(Random_salt_pepper_noise(prob=[0.02,0.1],p=0.2))

        augmentation2.append(RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1,p=0.6))
        augmentation2.append(RandomGrayscale(p=0.2))
        augmentation2.append(RandomGaussianBlur(sigma=[0.1,2.0],p=0.3))

        self.transform1 = transforms.Compose(augmentation1)
        self.transform2 = transforms.Compose(augmentation2)
        
    def __call__(self, data):
        img = data['image']
        img = Image.fromarray(img.astype("uint8"), "RGB")
        img = self.transform2(img)
        data['image'] = self.transform1(np.array(img))
        return data
class targetAugment_inplace(object):
    def __init__(self, **kwargs) -> None:
        augmentation1 = []
        augmentation2 = []
        
        augmentation1.append(Random_gaussian_noise(mean=0.1,sigma=[0.1,0.15],p=0.1))
        augmentation1.append(Random_salt_pepper_noise(prob=[0.02,0.1],p=0.1))

        augmentation2.append(RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1,p=0.2))
        augmentation2.append(RandomGrayscale(p=0.1))
        augmentation2.append(RandomGaussianBlur(sigma=[0.1,1],p=0.2))

        self.transform1 = transforms.Compose(augmentation1)
        self.transform2 = transforms.Compose(augmentation2)
        
    def __call__(self, data):
        img = data['image']
        img = Image.fromarray(img.astype("uint8"), "RGB")
        img = self.transform2(img)
        data['image'] = self.transform1(np.array(img))
        return data

class source_StrongAugment(object):
    def __init__(self, **kwargs) -> None:
        augmentation1 = []
        augmentation2 = []
        
        augmentation1.append(Random_gaussian_noise(mean=0.1,sigma=[0.1,0.15],p=0.2))
        augmentation1.append(Random_salt_pepper_noise(prob=[0.02,0.15],p=0.2))

        augmentation2.append(RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1,p=0.6))
        augmentation2.append(RandomGrayscale(p=0.2))
        augmentation2.append(RandomGaussianBlur(sigma=[0.1,2.0],p=0.4))

        self.transform1 = transforms.Compose(augmentation1)
        self.transform2 = transforms.Compose(augmentation2)
        
    def __call__(self, data):
        img = data['image']
        img = Image.fromarray(img.astype("uint8"), "RGB")
        img = self.transform2(img)
        data['strong_image'] = self.transform1(np.array(img))
        return data

class target_StrongAugment(object):
    def __init__(self, **kwargs) -> None:
        augmentation1 = []
        augmentation2 = []
        
        augmentation1.append(Random_gaussian_noise(mean=0.1,sigma=[0.1,0.15],p=0.1))
        augmentation1.append(Random_salt_pepper_noise(prob=[0.02,0.1],p=0.1))

        augmentation2.append(RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1,p=0.2))
        augmentation2.append(RandomGrayscale(p=0.1))
        augmentation2.append(RandomGaussianBlur(sigma=[0.1,1],p=0.2))

        self.transform1 = transforms.Compose(augmentation1)
        self.transform2 = transforms.Compose(augmentation2)
        
    def __call__(self, data):
        img = data['image']
        img = Image.fromarray(img.astype("uint8"), "RGB")
        img = self.transform2(img)
        data['strong_image'] = self.transform1(np.array(img))
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

class Random_gaussian_noise(object):
    def __init__(self,mean=0.1,sigma=[0.1,0.2],p=0.2):
        self.p = p
        self.mean=mean
        self.sigma=random.uniform(sigma[0],sigma[1])
    def __call__(self, data):
        if random.random() < self.p:
            # return gauss_noise(data,self.mean,self.sigma)
            return gaussian_noise(data,self.mean,self.sigma)
        return data
class Random_salt_pepper_noise(object):
    def __init__(self,prob=[0.02,0.2],p=0.2):
        self.p = p
        self.prob=prob
    def __call__(self, data):
        if random.random() < self.p:
            prob = random.uniform(self.prob[0],self.prob[1])
            return fast_salt_pepper_noise(data,prob)
        return data
def gauss_noise(img, mean=0.1, sigma=0.1):
    image = np.array(img / 255, dtype=float)  #
    # 
    noise = np.random.normal(mean, sigma, image.shape)
    out = image + noise  # 
    res_img = np.clip(out, 0.0, 1.0)
    res_img = np.uint8(res_img * 255.0) 
    
    return res_img

def gaussian_noise(image, mean=0.1, sigma=0.1):
    """
    :param image:original images
    :param mean:
    :param sigma: value biggger, noise bigger
    :return: result img
    """
    image = np.asarray(image / 255, dtype=np.float32) # 
    noise = np.random.normal(mean, sigma, image.shape).astype(dtype=np.float32)  # 
    output = image + noise  # 
    output = np.clip(output, 0.0, 1.0)
    output = np.uint8(output * 255.0)
    return output

def fast_salt_pepper_noise(image: np.ndarray, prob=0.02):
    """
    Randomly generate a 0-1 mask as salt and pepper noise
    :param image:
    :param prob: noise proportion
    :return:
    """
    image = add_uniform_noise(image, prob * 0.51, vaule=255)
    image = add_uniform_noise(image, prob * 0.5, vaule=0)
    return image
 
def add_uniform_noise(image: np.ndarray, prob=0.05, vaule=255):
    """
    Randomly generate a 0-1 mask as salt and pepper noise
    :param image:
    :param prob: noise proportion
    :param vaule: noise value
    :return:
    """
    h, w = image.shape[:2]
    noise = np.random.uniform(low=0.0, high=1.0, size=(h, w)).astype(dtype=np.float32)  # 
    mask = np.zeros(shape=(h, w), dtype=np.uint8) + vaule
    index = noise > prob
    mask = mask * (~index)
    output = image * index[:, :, np.newaxis] + mask[:, :, np.newaxis]
    output = np.clip(output, 0, 255)
    output = np.uint8(output)
    return output

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

# a = source_StrongAugment()
# # b = target_StrongAugment()
# # import time
# import os
# for name in os.listdir('imgs/'): 
#     for i in range(20):
#         img = cv2.imread(os.path.join('imgs',name))
#         img1 = {}    
#         # t = time.time()
#         img1['image'] = img
#         img1 = a(img1)
#         # cv2.imwrite('abc1.jpg',np.array(img1['image']))
#         cv2.imwrite(os.path.join('imgs',str(i)+name),np.array(img1['strong_image']))
        # print(time.time()-t)
# img = cv2.imread(os.path.join('imgs','1c_132.jpg'))
# img1 = {}    
# # t = time.time()
# img1['image'] = img
# img1 = a(img1)
# # cv2.imwrite('abc1.jpg',np.array(img1['image']))
# cv2.imwrite(os.path.join('imgs','1c_1321.jpg'),np.array(img1['strong_image']))