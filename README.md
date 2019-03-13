# Real-Time-Style-Transfer-in-QT

I trained Neural-Style-Transfer network and CycleGAN to implement different style transfer. Also, I implement gui interface via guizero and PyQt5 separately.

Because guizero is not efficient for real-time video stream, the implementation via guizero can only be used to real-time photo style transfer. PyQt5 is more convinent to show an opencv video stream.

This is the initial interface:

![](https://github.com/verBubble/Real-Time-Style-Transfer-in-QT/raw/master/images/1.png)

When you click a button like "Transfer_Picasso", it will show style image, original video image and transfer-image in real-time. Just like this:

![](https://github.com/verBubble/Real-Time-Style-Transfer-in-QT/raw/master/images/2.png)

The guizero display looks like this:

![](https://github.com/verBubble/Real-Time-Style-Transfer-in-QT/raw/master/images/3.png)

When you click a button like "CycleGAN_VanGogh", it will look like this:

![](https://github.com/verBubble/Real-Time-Style-Transfer-in-QT/raw/master/images/4.png)

## Preparation

Python 3.6 and Pytorch 0.4.1

guizero and PyQt5

opencv 3.4.3

To install PyQt5, run:

```python
pip install PyQt5
pip install PyQt5-tools
```

To install guizero, run:

```
pip install guizero 
```

And also, in my implementation, GPU is needed. If you want to run it on CPU, just remove codes like `xxx.cuda()`

## Test 

Download the weights first.

```
wget https://github.com/verBubble/Real-Time-Style-Transfer-in-QT/releases/download/style-transfer/GodBearer.pth

wget https://github.com/verBubble/Real-Time-Style-Transfer-in-QT/releases/download/style-transfer/picasso.pth
```

Save these 2 weights into '/Neural_Style/checkpoints'.

```
wget https://github.com/verBubble/Real-Time-Style-Transfer-in-QT/releases/download/CycleGAN/monet.pth

wget https://github.com/verBubble/Real-Time-Style-Transfer-in-QT/releases/download/CycleGAN/VanGogh.pth
```

Save these 2 weights into '/CycleGAN/outputs'.

Run guizero code:

```
python display.py
```

Run PyQt5 code:

```
python display_pyqt.py
```

## One more thing 

My Neural_Style network is based on this repo: https://github.com/chenyuntc/pytorch-book

My CycleGAN network is based on this repo: https://github.com/aitorzip/PyTorch-CycleGAN

If you want to train by yourself, follow instructions of these two repo.