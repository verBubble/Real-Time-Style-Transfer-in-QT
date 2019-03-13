import sys
import cv2
import numpy as np
import torch
import argparse
import os
from imutils import paths
from PIL import Image

from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QApplication, QHBoxLayout, QVBoxLayout)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSignal, Qt

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable

# CycleGAN
from CycleGAN.models import Generator
from CycleGAN.datasets import ImageDataset

#Neural_Style
from Neural_Style.transformer_net import TransformerNet 

# define global variable model and trm, when thread is starting, it will use the model directly.
# When clicking a button, model and trm will be changed.
class G:
    # 0 for no model, 1 for CycleGAN, 2 for Neural_Style
    ModelIndex = 0
    Model = None
    Trm = None

# define model init
def CycleGAN_init(weight_path):
    model = Generator(3, 3)
    model.load_state_dict(torch.load(weight_path))
    model.cuda()
    model.eval()
    G.Model = model

    trm = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Lambda(lambda x: x*255)  Neural_Style
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # cycleGAN
    ])
    G.Trm = trm


def NeuralStyle_init(weight_path):
    model = TransformerNet()
    model.load_state_dict(torch.load(weight_path))
    model.cuda()
    model.eval()
    G.Model = model

    trm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255)
    ])
    G.Trm = trm



# define a function to convert image from opencv to qtformat
def cv2QtFormat(cv_image):
    rgbImage = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
    convertToQtFormat = QPixmap.fromImage(convertToQtFormat)
    p = convertToQtFormat.scaled(256, 256)
    return p

# define transfer function
def Transfer(img):

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
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
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img



# define thread
class Thread(QThread):
    changePixmap2 = pyqtSignal(QPixmap)
    changePixmap3 = pyqtSignal(QPixmap)

    def __init__(self, parent = None):
        QThread.__init__(self, parent = parent)

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, img = cap.read()
            if ret:
                if G.ModelIndex != 0:
                    p = Transfer(img)
                    img = cv2QtFormat(img)
                    p = cv2QtFormat(p)
                    self.changePixmap2.emit(img)
                    self.changePixmap3.emit(p)


# define window
class Window(QWidget):

    Dic_God = {'modelIndex': 2, 'weight_path': "Neural_Style/checkpoints/GodBearer.pth",
    'image_path': "Neural_Style/outputs/style.jpg"}
    Dic_Pic = {'modelIndex': 2, 'weight_path': "Neural_Style/checkpoints/picasso.pth",
    'image_path': "Neural_Style/outputs/picasso.jpg"}
    Dic_Van = {'modelIndex': 1, 'weight_path': "CycleGAN/outputs/VanGogh.pth",
    'image_path': "CycleGAN/outputs/VanGogh.jpg"}
    Dic_Mon = {'modelIndex': 1, 'weight_path': "CycleGAN/outputs/monet.pth",
    'image_path': "CycleGAN/outputs/monet.jpg"}


    def __init__(self):
        super().__init__()
        self.Init_UI()

    def Init_UI(self):

        self.setGeometry(300,300,800,500)
        self.setWindowTitle('Style_Transfer')

        # define text 
        text4 = QLabel(self)
        text4.setText("You can click buttons below to change style.")

        # define 4 buttons
        button1 = QPushButton('Transfer_Godbearer', self)
        button2 = QPushButton('Transfer_Picasso', self)
        button3 = QPushButton('CycleGAN_VanGogh', self)
        button4 = QPushButton('CycleGAN_Monet', self)

        # adjust locations of buttons
        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addWidget(text4)
        vbox.addWidget(button1)
        vbox.addWidget(button2)
        vbox.addWidget(button3)
        vbox.addWidget(button4)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addLayout(vbox)

        self.setLayout(hbox)

        # connect buttons with function
        button1.clicked.connect(lambda: self.Creat_Model(self.Dic_God))
        button2.clicked.connect(lambda: self.Creat_Model(self.Dic_Pic))
        button3.clicked.connect(lambda: self.Creat_Model(self.Dic_Van))
        button4.clicked.connect(lambda: self.Creat_Model(self.Dic_Mon))

        # define text to interpret images
        text1 = QLabel(self)
        text2 = QLabel(self)
        text3 = QLabel(self)

        text1.setText('style_image')
        text2.setText('video_image')
        text3.setText('Transfer_image')

        text1.move(115, 260)
        text2.move(385, 260)
        text3.move(655, 260)

        # define 3 images
        self.image1 = QLabel(self) #style image
        image2 = QLabel(self)   #video image
        image3 = QLabel(self)   #transfer image

        self.image1.resize(256, 256)
        image2.resize(256, 256)
        image3.resize(256, 256)

        self.image1.move(1, 3)
        image2.move(270, 3)
        image3.move(530, 3)

        # When first opening it, show 3 default images
        pixmap = QPixmap('Neural_Style/outputs/style.jpg')
        pixmap = pixmap.scaled(256, 256)
        self.image1.setPixmap(pixmap)
        pixmap = QPixmap('Neural_Style/outputs/17.png')
        pixmap = pixmap.scaled(256, 256)
        image2.setPixmap(pixmap)
        pixmap = QPixmap('Neural_Style/outputs/out.png')
        pixmap = pixmap.scaled(256, 256)
        image3.setPixmap(pixmap)

        # define a thread
        self.th = Thread(self)
        self.th.changePixmap2.connect(image2.setPixmap)
        self.th.changePixmap3.connect(image3.setPixmap)
        self.th.start()

        self.show()

    def Creat_Model(self, dic):
        torch.cuda.empty_cache()
        if dic['modelIndex'] == 1:
            CycleGAN_init(dic['weight_path'])
        if dic['modelIndex'] == 2:
            NeuralStyle_init(dic['weight_path'])

        G.ModelIndex = dic['modelIndex']
        pixmap = QPixmap(dic['image_path'])
        pixmap = pixmap.scaled(256, 256)
        self.image1.setPixmap(pixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Window()
    app.exit(app.exec_())