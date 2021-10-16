#!/bin/sh

model='vggface2/20180402-114759.pb'
image1='images/iuchi005.jpg'
image2='images/sentaro003.jpg'
image3='images/sentaro005.jpg'
image4='images/sentaro007.jpg'

# /opt/anaconda3/envs/facenet/bin/python3 src/compare.py ${model} ${image1} ${image2} --image_size 160 --margin 32 --gpu_memory_fraction 0

#/opt/anaconda3/envs/facenet/bin/python3 src/compare_camera.py ${model} ${image1} ${image2} --image_size 160 --margin 32 --gpu_memory_fraction 0

/opt/anaconda3/envs/facenet/bin/python3 src/face_recognition.py ${model} ${image1} ${image2} ${image3} ${image4} --image_size 160 --margin 32 --gpu_memory_fraction 0
