import os
import re
import pydicom

def change_dirname (root, patient, directory):
    img_dir = root+'/'+patient+'/ST0/'+directory+'/'
    img = pydicom.dcmread (img_dir+'IM0')
    new_name = re.sub (' ', '_', img.SeriesDescription)
    new_dir = re.sub ('/ST0', '', re.sub (directory+'/$', new_name, img_dir))
    for one_img in os.listdir (img_dir):
        os.rename (img_dir+'/'+one_img, img_dir+'/'+one_img+'.dcm')
    os.rename (img_dir, new_dir)

def change_dirname_patient (root, patient):
    root_dir = root+'/'+patient+'/ST0/'
    all_dir = sorted (os.listdir (root_dir))
    for i in all_dir:
        change_dirname (root, patient, i)

def change_dirname_all (root):
    for one_patient in sorted (os.listdir (root)):
        change_dirname_patient (root, one_patient)

if __name__ == '__main__':
    change_dirname_all ('/mnt/d/DICOMDIU/')
