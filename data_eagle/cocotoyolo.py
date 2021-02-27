
from pycocotools.coco import COCO
coco = COCO(r'D:\INTERVIEWS\eagleview\trainval_n\trainval\annotations\bbox-annotations.json')

ImgIds = coco.getImgIds()
CatIds = coco.getCatIds()
#%%
correct_ims = []

for img_id in ImgIds:
    print(img_id)
    AnnIds = coco.getAnnIds(img_id, CatIds)
    anns = coco.loadAnns(AnnIds)
    
    file_name = coco.loadImgs(img_id)[0]['file_name']
    wid = coco.loadImgs(img_id)[0]['width']
    hig = coco.loadImgs(img_id)[0]['height']
    
    file_txt = file_name.split('.')[0]+'.txt'
    
    an_list = []
    for an in anns:
        l_0 = an['category_id']-1
        l_1 = max(an['bbox'][0]+an['bbox'][2]/2,1)/wid
        l_2 = max(an['bbox'][1]+an['bbox'][3]/2,1)/hig
        l_3 = max(an['bbox'][2],1)/wid
        l_4 = max(an['bbox'][3],1)/hig
        
        an_yolo = str(l_0)+' '+str(l_1)+' '+str(l_2)+' '+str(l_3)+' '+str(l_4)
        an_list.append(an_yolo)
    
    if l_1 < 1 and l_2 < 1 and l_3 < 1 and l_4 < 1:
        correct_ims.append(img_id)
    else:
        print('Wrong', img_id)
        

    with open(file_txt, 'w') as f:
        f.writelines('\n'.join(an_list))
#%%
import random
import os

img_names = []
for img_id in correct_ims:
    file_name = coco.loadImgs(img_id)[0]['file_name']
    img_names.append(file_name)
    
img_names = ['data_eagle/obj/'+im for im in img_names]

random.shuffle(img_names)
train_split = img_names[:2100]
test_split = img_names[2100:]

with open('train.txt', 'w') as f:
    f.writelines('\n'.join(train_split))

with open('test.txt', 'w') as f:
    f.writelines('\n'.join(test_split))

with open('train_all.txt', 'w') as f:
    f.writelines('\n'.join(img_names))

#%%
import matplotlib.pyplot as plt
import cv2
import os
from random import randint

def test_cords(n, root):
    for img_id in ImgIds[:n]:
        file_name = coco.loadImgs(img_id)[0]['file_name']    
        pa = os.path.join(root, file_name)
        img = cv2.imread(pa)
        AnnIds = coco.getAnnIds(img_id, CatIds)
        anns = coco.loadAnns(AnnIds)
        
        for an in anns:
            x = an['bbox'][0]
            y = an['bbox'][1]
            w = an['bbox'][2]
            h = an['bbox'][3]
        
            cv2.rectangle(img, (x,y), (x+w, y+h), (randint(0, 255),randint(0, 255),randint(0, 255)), 2)
            print(x,y,w,h)
        
        cv2.imwrite(file_name, img)
        print(file_name)
#%%
test_cords(20,r'D:\INTERVIEWS\eagleview\eagle_yolo\data_eagle\obj')
