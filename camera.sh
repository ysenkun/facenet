#!/bin/sh

model='vggface2/20180402-114759.pb'

#facenet
#/opt/anaconda3/envs/facenet/bin/python3 src/camera_face.py ${model} --image_size 160 --margin 32 --gpu_memory_fraction 0.5

#facenet_mask
/opt/anaconda3/envs/facenet_mask/bin/python3 src/camera_face.py ${model} --image_size 160 --margin 32 --gpu_memory_fraction 0.5
