import cv2
import numpy as np
import torch
import random

def horiz_flip(batch, batch_size):
    """
    Inputs:
     - batch: tuple containing data (shape = [batchsize, 48, 48]) and targets (shape = [batchsize, 1])
     - batch_size: batchsize
    Returns:
     - (batchsize, 48, 48) tensor containing batch of images
    """    
    flipped_imgs = []
    for i in range(batch_size):
        try:   #Try Catch in order not to cause error on the last batch which may not be complete
            img_tensor = batch[0][i]
            flipped = torch.from_numpy(cv2.flip(np.array(img_tensor), 1))
            flipped_imgs.append(flipped)
        except: 
            break
    
    flipped_imgs_tensor = torch.stack(flipped_imgs)
    return flipped_imgs_tensor 
    

def zoom(batch, batch_size):
    """
    Inputs:
     - batch: tuple containing data (shape = [batchsize, 48, 48]) and targets (shape = [batchsize, 1])\
     - batch_size: batchsize
    Returns:
     - (batchsize, 48, 48) tensor containing batch of images
    """
    zoom_factor = 1.25
    zoomed_imgs = []
    for i in range(batch_size):
        try:   #Try Catch in order not to cause error on the last batch which may not be complete
            img_tensor = batch[0][i]
            img_np = np.array(img_tensor, dtype="uint8")
            enlarged_size = int(48*zoom_factor)
            enlarged = cv2.resize(img_np, (enlarged_size, enlarged_size))

            cropped = enlarged[int(enlarged_size/2)-24:int(enlarged_size/2)+24, int(enlarged_size/2)-24:int(enlarged_size/2)+24]

            zoomed_imgs.append(torch.from_numpy(cropped))
        except: 
            break
    zoomed_imgs_tensor = torch.stack(zoomed_imgs)
    return zoomed_imgs_tensor


def rotate(batch, batch_size):
    """
    Inputs:
     - batch: tuple containing data (shape = [batchsize, 48, 48]) and targets (shape = [batchsize, 1])\
     - batch_size: batchsize
    Returns:
     - (batchsize, 48, 48) tensor containing batch of images
     - (batchsize, 1) tensor containing batch of labels
    """
    angle = 15    #degrees
    rotated_imgs = []
    for i in range(batch_size):
        try:   #Try Catch in order not to cause error on the last batch which may not be complete
            img_tensor = batch[0][i]
            img_np = np.array(img_tensor, dtype="uint8")
            img_center = tuple(np.array(img_np.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(img_center, angle, 1.0)
            rotated = cv2.warpAffine(img_np, rot_mat, img_np.shape[1::-1], flags=cv2.INTER_LINEAR)
            rotated_imgs.append(torch.from_numpy(rotated))
        except: 
            break
    
    rotated_imgs_tensor = torch.stack(rotated_imgs)
    return rotated_imgs_tensor 

def horiz_flip_and_rotate(batch, batch_size):
    """
    Inputs:
     - batch: tuple containing data (shape = [batchsize, 48, 48]) and targets (shape = [batchsize, 1])\
     - batch_size: batchsize
    Returns:
     - (batchsize, 48, 48) tensor containing batch of images
     - (batchsize, 1) tensor containing batch of labels
    """
    angle = -10    #degrees
    augmented_imgs = []

    for i in range(batch_size):
        try:   #Try Catch in order not to cause error on the last batch which may not be complete
            img_tensor = batch[0][i]
            img_np = np.array(img_tensor, dtype="uint8")
            flipped = cv2.flip(img_np, 1)

            img_center = tuple(np.array(flipped.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(img_center, angle, 1.0)
            rotated = cv2.warpAffine(flipped, rot_mat, flipped.shape[1::-1], flags=cv2.INTER_LINEAR)

            augmented_imgs.append(torch.from_numpy(rotated))
        except: 
            break
    
    augmented_imgs_tensor = torch.stack(augmented_imgs)
    return augmented_imgs_tensor
    
def brightness(batch, batch_size):
    """
    Inputs:
     - batch: tuple containing data (shape = [batchsize, 48, 48]) and targets (shape = [batchsize, 1])\
     - batch_size: batchsize
    Returns:
     - (batchsize, 48, 48) tensor containing batch of images
     - (batchsize, 1) tensor containing batch of labels
    """
    multiplier = 0.7
    augmented_imgs = []
    for i in range(batch_size):
        try:   #Try Catch in order not to cause error on the last batch which may not be complete
            img_tensor = batch[0][i]
            augmented = multiplier * img_tensor
            augmented_imgs.append(augmented)
        except: 
            break
    
    augmented_imgs_tensor = torch.stack(augmented_imgs)
    return augmented_imgs_tensor 

def translate(batch, batch_size):
    """
    Inputs:
     - batch: tuple containing data (shape = [batchsize, 48, 48]) and targets (shape = [batchsize, 1])\
     - batch_size: batchsize
    Returns:
     - (batchsize, 48, 48) tensor containing batch of images
     - (batchsize, 1) tensor containing batch of labels
    """
    max_translation = 0.15
    augmented_imgs = []
    for i in range(batch_size):
        try:   #Try Catch in order not to cause error on the last batch which may not be complete
            img_tensor = batch[0][i]
            img_np = np.array(img_tensor, dtype="uint8")
            random_width = int((random.random()*(max_translation-0.05) + 0.05) * (1 if random.random() < 0.5 else -1) * img_np.shape[1])
            random_height = int((random.random()*(max_translation-0.05) + 0.05) * (1 if random.random() < 0.5 else -1) * img_np.shape[0])
            T = np.float32([[1, 0, random_width], [0, 1, random_height]])
            augmented = cv2.warpAffine(img_np, T, (img_np.shape[1], img_np.shape[0]))
            augmented_imgs.append(torch.from_numpy(augmented))
        except: 
            break
    
    augmented_imgs_tensor = torch.stack(augmented_imgs)
    return augmented_imgs_tensor

def scale(batch, batch_size):
    """
    Inputs:
     - batch: tuple containing data (shape = [batchsize, 48, 48]) and targets (shape = [batchsize, 1])\
     - batch_size: batchsize
    Returns:
     - (batchsize, 48, 48) tensor containing batch of images
     - (batchsize, 1) tensor containing batch of labels
    """
    return None