from guizero import App, PushButton, Picture, Text

import cv2
import numpy as np
import torch
import argparse
import sys
import os
from imutils import paths
from PIL import Image

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable

# CycleGAN
from CycleGAN.models import Generator
from CycleGAN.datasets import ImageDataset

#Neural_Style
from Neural_Style.transformer_net import TransformerNet 

# We will have 3 images:A, B and C. A is style image, B is video image, C is after_transfer image.
# When displaying in this gui, images is 256*256. When in glasses, images will be larger.

# ModelIndex
# define a variable to know which model(CycleGAN or Neural_style) is occupying GPU
# 0 for no model, 1 for CycleGAN, 2 for Neural_Style
# if_open
# define a viriable to know if the camera is open, 0 for close, 1 for open

class G:
    if_open = 0
    ModelIndex = 0
    cap = None


def CycleGAN_init(weight_path):
    model = Generator(3, 3)
    model.load_state_dict(torch.load(weight_path))
    model.cuda()
    model.eval()

    trm = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Lambda(lambda x: x*255)  Neural_Style
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # cycleGAN
    ])

    return model,trm

def NeuralStyle_init(weight_path):
    model = TransformerNet()
    model.load_state_dict(torch.load(weight_path))
    model.cuda()
    model.eval()

    trm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255)
    ])

    return model,trm 

def open_camera(index = 0):
    G.cap = cv2.VideoCapture(index)
    G.if_open = 1


def reset_img(picture, PILimage):
    #if PILimage == 1:
    #    picture.set("temp/before.png")
    #if PILimage ==2:
    #    picture.set("temp/after/png")
    picture.value = PILimage

def display_video():

    while(True):
        success,img = G.cap.read()
        #img = Image.fromarray(img).resize((512,512))
        #img = np.array(img)
        #cv2.imshow("before",img)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_show = Image.fromarray(img)
        # img is RGB for model processing
        img = Image.fromarray(img).resize((256,256))

    

        img = G.Trm(img).cuda()
        # img is RGB
        t_img = G.Model(img.unsqueeze(0)).data.squeeze(0).cpu()

        #t_img /=255  style transfer
        t_img = 0.5*(t_img + 1.0)
        t_img[t_img > 1] = 1
        t_img[t_img < 0] = 0
        img = transforms.ToPILImage()(t_img)
        img_show2 = transforms.ToPILImage()(t_img)
        img = img.resize((256,256))
        img = np.array(img)
        # img is RGB, but cv2 is showing BGR...
        #cv2.imshow("after",cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        B.repeat(1000, reset_img, args = [B, img_show])
        C.repeat(1000, reset_img, args = [C, img_show2])


def display_photo():

    success,img = G.cap.read()
    img = Image.fromarray(img).resize((256,256))
    img = np.array(img)
    #cv2.imshow("before",img)

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_show = Image.fromarray(img)
    # img is RGB for model processing
    img = Image.fromarray(img).resize((256,256))

    img = G.Trm(img).cuda()
    # img is RGB
    t_img = G.Model(img.unsqueeze(0)).data.squeeze(0).cpu()
    if G.ModelIndex==2:
        t_img /=255  
    if G.ModelIndex==1:
        t_img = 0.5*(t_img + 1.0)
    t_img[t_img > 1] = 1
    t_img[t_img < 0] = 0
    img = transforms.ToPILImage()(t_img)
    img_show2 = transforms.ToPILImage()(t_img)
    # img is RGB, but cv2 is showing BGR...
    #cv2.imshow("after",cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    B.value = img_show
    C.value = img_show2

def Transfer(modelIndex, weight_path, image_path):

    # if G.ModelIndex!=1:
    #    torch.cuda.empty_cache()
    #    G.Model, G.Trm = CycleGAN_init("CycleGAN/outputs/VanGogh.pth")
    #else:
    #    G.Model.load_state_dict(torch.load("CycleGAN/outputs/VanGogh.pth"))
    
    torch.cuda.empty_cache()

    if(modelIndex == 1):
    	G.Model, G.Trm = CycleGAN_init(weight_path)
    	G.ModelIndex = 1
    else:
    	G.Model, G.Trm = NeuralStyle_init(weight_path)
    	G.ModelIndex = 2

    if G.if_open == 0:
        open_camera()

    A.value = image_path

    display_photo()


app = App(layout = "grid", title = "Style_Transfer", height = 550, width = 1000)
# define 5 buttons
button1 = PushButton(app, text = "Transfer_Godbearer", command = Transfer, 
	args = [2, "Neural_Style/checkpoints/GodBearer.pth", "Neural_Style/outputs/style.jpg"], 
	grid = [2,2], align = "right")
button2 = PushButton(app, text = "Transfer_Picasso", command = Transfer, 
	args = [2, "Neural_Style/checkpoints/picasso.pth", "Neural_Style/outputs/picasso.jpg"],
	grid = [2,3], align = "right")
button3 = PushButton(app, text = "CycleGAN_VanGogh", command = Transfer, 
	args = [1, "CycleGAN/outputs/VanGogh.pth", "CycleGAN/outputs/VanGogh.jpg"],
	grid = [2,4], align = "right")
button4 = PushButton(app, text = "CycleGAN_Monet", command = Transfer, 
	args = [1, "CycleGAN/outputs/monet.pth", "CycleGAN/outputs/monet.jpg"],
	grid = [2,5], align = "right")
button5 = PushButton(app, text = "Take a photo", command = display_photo, grid = [2,6], align = "right")

# define 3 texts
text1 = Text(app, text="style_image", grid = [0,0])
text2 = Text(app, text="video_image", grid = [1,0])
text3 = Text(app, text="transfer_image", grid = [2,0])
text4 = Text(app, text="You can click buttons on the right", grid = [0,3])
text5 = Text(app, text="to change the images", grid = [1,3], align = "left")

# define 3 images
A = Picture(app, image = "Neural_Style/outputs/style.jpg", grid = [0,1], align = "left", height = 256, width = 256)
B = Picture(app, image = "Neural_Style/outputs/17.png", grid = [1,1], align = "left", height = 256, width = 256)
C = Picture(app, image = "Neural_Style/outputs/out.png", grid = [2,1], align = "left", height = 256, width = 256)

app.display()