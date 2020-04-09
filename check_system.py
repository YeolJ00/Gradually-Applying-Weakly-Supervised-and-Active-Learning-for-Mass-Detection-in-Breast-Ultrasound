import os
import pickle
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree

def parse_ws(filename):
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        if obj_struct['name'] == '__background__':
            continue
        objects.append(obj_struct)
    return objects

dataset = 'al_train'


if dataset == 'al_train':
    annopath = os.path.join('./data/SNUBH_BUS/Annotations/','{:s}_AL.xml')
else:
    annopath = os.path.join('./data/SNUBH_BUS/Annotations/','{:s}.xml')
imagesetfile = os.path.join('./data/SNUBH_BUS/ImageSets/Main/','{}.txt'.format(dataset))
save = os.path.join('./analysis_{}.txt'.format(dataset))

with open(imagesetfile, 'r') as f:
    lines = f.readlines()
imagenames = [x.strip() for x in lines]

benign = 0
malignant = 0
with open(save,'w') as f:
    for i, imagename in enumerate(imagenames):
        objects = parse_ws(annopath.format(imagename))
        if objects != []:
            f.write('{}\t{}\n'.format(imagename,objects[0]['name']))
            if objects[0]['name'] == 'benign':
                benign += 1
            else:
                malignant +=1
    f.write('total: {}\tbenign: {}\tmalignant: {}\n'.format(benign+malignant,benign,malignant))