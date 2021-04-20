import os
from os.path import *

relabel_dict = {
    1:9,
    2:8,
    3:7,
    4:6,
    5:5,
    6:4,
    7:3,
    8:2,
    9:7,
    10:2,
    11:6
}

def foo(PATH, TOPATH):
    with open(PATH, 'r') as f:
        image_list = f.readlines()

    new_f = open(TOPATH, 'a')
    for row in image_list:
        row = row.strip().split(" ")
        name,label = row[0],int(row[1])
        new_f.writelines(name+' '+str(relabel_dict[label])+'\n')
    new_f.close()

def patch_op(PATH_LIST):
    for path in PATH_LIST:
        to_path = path.split('.')[0]+'_relabel.txt'
        foo(path,to_path)


patch_op(['batch_2_train_kfold.txt','batch_2_crossval_kfold.txt','batch_2_test_kfold.txt','batch_3_train_kfold.txt',\
    'batch_3_crossval_kfold.txt','batch_3_test_kfold.txt'])

