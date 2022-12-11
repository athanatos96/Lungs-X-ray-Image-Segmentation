import os
import cv2
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# Open the csv and corresponding img
def import_folder_dataset(root_path, resized_side=(1024,1024)):
    white = [255,255,255]
    
    img_l = []
    mask_l = []
    new_mask_l = []
    
    
    
    # Build the path for the images and the mask
    images = os.path.join(root_path, "Images")
    mask = os.path.join(root_path, "Masks")
    
    # List all the images names
    images_list = os.listdir(images)
    
    # Iterate the images names
    #for im_name in images_list:
    for im_name in tqdm(images_list, total=len(images_list)):
        # build the path for the img and the corresponding mask
        img_path = os.path.join(images, im_name)
        mask_path = os.path.join(mask, im_name)
        
        img = cv2.imread(img_path)
        ma = cv2.imread(mask_path)
        
        
        img = cv2.resize(img, resized_side)
        ma = cv2.resize(ma, resized_side)
        
        # Convert the mask to 2 channels only, each channel correspond to 1 class
        #new_ma = np.array([[[0,1] if (y & white).any() else [1,0] for y in x ] for x in ma])
        new_ma = cv2.cvtColor(ma, cv2.COLOR_BGR2GRAY)
        new_ma = (new_ma/255).astype(np.uint8)
        
        
        img_l.append(img)
        mask_l.append(ma)
        new_mask_l.append(new_ma)
        
        
    return( (np.array(img_l), np.array(mask_l), np.array(new_mask_l)) )
    
# Create the dataset object
class Data(Dataset):
    def __init__(self, X_element, m_element, y_element, transform):
        # Save images
        self.X = X_element
        # Save mask images
        self.m = m_element
        # Save segmentation classes
        self.y = torch.from_numpy(y_element).type(torch.LongTensor)
        
        self.transform = transform
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        
        item = self.X[index]
        item = self.transform(item)
        
        return item, self.m[index], self.y[index]


def calculate_IoU(pred, label):
    
    overlap = torch.logical_and(pred, label)
    union = torch.logical_or(pred, label)
    
    iou = torch.sum(overlap, dim = [-1,-2] ) / torch.sum(union, dim = [-1,-2] )
    
    return iou

def calculate_Dice(pred, label):
    
    overlap = torch.logical_and(pred, label)
    denominator = torch.sum(pred, dim = [-1,-2] ) + torch.sum(label, dim = [-1,-2] )
    
    
    iou = torch.sum(overlap, dim = [-1,-2] ) / denominator
    
    return 2*iou
    

def plot_loss_accuracy(train_loss, val_loss, train_IoU, val_IoU):
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,6))

    fig.suptitle('Horizontally stacked subplots')
    ax1.plot(train_loss, label="Train_loss")
    ax1.plot(val_loss, label="Validation_loss")
    ax1.title.set_text("Loss")
    ax1.legend(loc="best")

    ax2.plot(train_IoU, label="train_IoU")
    ax2.plot(val_IoU, label="Validation_IoU")
    ax2.title.set_text("Accuracy")
    ax2.legend(loc="best")

    plt.show()
    
    
    # create figure and axis objects with subplots()
    fig,ax = plt.subplots(figsize=(15,6))
    # make a plot
    ax.plot(train_loss, color="red", marker="o", label="Train_loss")
    ax.plot(val_loss, color="orange", marker="o", label="Validation_loss")
    # set x-axis label
    ax.set_xlabel("Epoch", fontsize = 14)
    # set y-axis label
    ax.set_ylabel("Loss Function",
                  color="red",
                  fontsize=14)


    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(train_IoU,color="blue",marker="^", label="train_IoU")
    ax2.plot(val_IoU,color="green",marker="^", label="Validation_IoU")

    ax2.set_ylabel("Accuracy",color="blue",fontsize=14)

    ax.legend(loc="center left")
    ax2.legend(loc="center right")
    plt.show()


# Save model Checkpoint    
def save_model(epochs, time, model, optimizer, criterion, path):
    """
    Function to save the trained model to disk.
    """
    # Remove the last model checkpoint if present.
    torch.save({
                'epoch': epochs+1,
                'time': time,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, path)
    
def save_metrics(train_loss, val_loss, train_IoU, val_IoU, path):
    # Method to save the results as a csv. Method by Alejandro C Parra Garcia
    dict = {'train_loss': train_loss, 'val_loss': val_loss, 'train_IoU': train_IoU, 'val_IoU': val_IoU}  
    df = pd.DataFrame(dict)  
    df.to_csv(path)

    
    
    

def make_predictions(loader, model, invTransforamtion, device):
    model.eval()

    image_list = []
    real_mask = []
    predictions_list = []
    accumulatedIoU = 0
    accumulatedDice = 0
    total = 0
    with torch.no_grad():
        for bi, data in tqdm(enumerate(loader), total=len(loader)):

            # Use model_bbox model to get the new cropped images basen on CAM
            images = data[0].to(device)# use the gpu
            labels = data[2].to(device)# use the gpu

            outputs = model(images)

            # calculate IoU
            predictions = torch.argmax(outputs, dim=1)
            total += labels.size(0)

            iioouu = calculate_IoU(predictions, labels)
            accumulatedIoU += iioouu.sum().item()
            
            
            dicee = calculate_Dice(predictions, labels)
            accumulatedDice += dicee.sum().item()
            
            # Invert the img transformation, and reorder to an Img
            im = np.moveaxis(invTransforamtion(data[0][0]).numpy(), 0, -1)
            
            image_list.append(im)
            real_mask.append(data[1][0].numpy())
            predictions_list.append(predictions[0].detach().cpu().numpy())

    mean_IoU = accumulatedIoU/total
    mean_dice = accumulatedDice/total
    return(mean_IoU, mean_dice, image_list, real_mask, predictions_list)