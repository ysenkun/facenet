#!/bin/sh

model='vggface2/20180402-114759.pb'
image1='images/iuchi001.jpg'
image2='images/sentaro001.jpg'
file='image'

/opt/anaconda3/envs/facenet/bin/python3 src/create_register.py ${model} ${image1} ${image2} --image_size 160 --margin 32 --gpu_memory_fraction 0
