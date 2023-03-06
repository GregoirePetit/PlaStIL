from multiprocessing import Pool
import random
import shutil
import torch
import sys
import os
from sklearn.svm import LinearSVC
from configparser import ConfigParser
import numpy as np
import pandas as pd

from MyFeatureFolder import L4FeaturesDataset

from multiprocessing import Pool, cpu_count, Array

if len(sys.argv) != 3:
    print('Arguments: config, last')
    sys.exit(-1)
cp = ConfigParser()
with open(sys.argv[1]) as fh:
    cp.read_file(fh)

cp = cp['config']
nb_classes = int(cp['nb_classes'])
normalization_dataset_name = cp['dataset']
first_batch_size = int(cp["first_batch_size"])
P = first_batch_size
state_size = P
S = nb_classes // P
batch_size = int(cp["batch_size"])
regul = float(cp["regul"])
toler = float(cp["toler"])
feat_root = cp["feat_root"]
classifiers_root = cp["classifiers_root"]
list_root = cp["list_root"]
model_root = cp["model_root"]
random_seed = int(cp["random_seed"])
num_workers = int(cp['num_workers'])

datasets_mean_std_file_path = cp['mean_std']
last = sys.argv[2]
images_list_dir = os.path.join(list_root, normalization_dataset_name)

destination_dir = os.path.join(feat_root, normalization_dataset_name, "seed"+str(random_seed),"b"+str(first_batch_size))
destination_dir_last = os.path.join(feat_root, normalization_dataset_name, "seed"+str(random_seed),"b"+str(first_batch_size),last)
classifiers_dir = os.path.join(classifiers_root, normalization_dataset_name, "seed"+str(random_seed),"b"+str(first_batch_size))
classifiers_dir_last = os.path.join(classifiers_root, normalization_dataset_name, "seed"+str(random_seed),"b"+str(first_batch_size),last)
print('normalization dataset name = ' + str(normalization_dataset_name))



def normalize_train_features(il_dir,state_id,state_size):
    feats_libsvm = []
    min_pos = (state_id) * state_size
    max_pos = (state_id+1) * state_size
    current_range_classes = range(min_pos,max_pos)
    class_list = list(range(nb_classes))
    
    train_dataset = L4FeaturesDataset(il_dir, range_classes=current_range_classes)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, num_workers=70, pin_memory=True)
    matrix_mix = list(iter(train_loader))
    matrix = matrix_mix[0][0]
    classes = list(matrix_mix[0][1])
    for i in range(1,len(matrix_mix)):
        matrix = torch.vstack((matrix,torch.nn.functional.normalize(matrix_mix[i][0], p=2, dim=1)))
        classes.extend(matrix_mix[i][1])
    feats_libsvm = matrix.numpy()
    
    return (np.array(classes),feats_libsvm)
    
    
""" MAIN """
if __name__ == '__main__':
    for state_id in range(S):
        curri = str(state_id+1)
        print()
        print("*"*50)
        print("State",curri, "of", S)
        print("*"*50)
        for considered_batch in ['batch1','batch'+str(curri)]:
            features_dir = destination_dir_last
            classif_dir = classifiers_dir_last
            if considered_batch == 'batch1':
                features_dir = destination_dir
                classif_dir = classifiers_dir
            il_dir = os.path.join(features_dir,considered_batch,"train")
            svms_dir =  os.path.join(classif_dir, considered_batch)
            norm_type = 'l2'
            min_pos = (state_id) * state_size
            max_pos = (state_id+1) * state_size
            current_range_classes = range(min_pos,max_pos)
            to_check = any([not os.path.exists(os.path.join(svms_dir,str(crt_id)+".model")) for crt_id in range(min_pos,max_pos)])
            if not to_check:
                print("SVM already created for range",[os.path.join(svms_dir,str(crt_id)+".model") for crt_id in [min_pos,max_pos-1]])
            else:
                print("Preparing state",curri, "of", S)
                print("Root path:", il_dir)
                def decompose_class(n):
                    file_path = os.path.join(il_dir, str(n))
                    if os.path.exists(file_path):
                        try:
                            os.makedirs(file_path+'_decomposed', exist_ok=True)
                            compteur = 0
                            with open(file_path, 'r') as f:
                                for line in f:
                                    with open(os.path.join(file_path+'_decomposed', str(compteur)), 'w') as f2:
                                        f2.write(line)
                                    compteur += 1
                        except:
                            pass
                with Pool() as p:
                    p.map(decompose_class, current_range_classes)
                y_true,norm_feats = normalize_train_features(il_dir,state_id,state_size)
                print("Cleaning state",curri, "of", S)

                def decompose_class(n):
                    file_path = os.path.join(il_dir, str(n))
                    try:
                        shutil.rmtree(file_path+'_decomposed')
                    except:
                        pass

                with Pool() as p:
                    p.map(decompose_class, range(nb_classes))
                df = pd.DataFrame(norm_feats, columns=['feat'+str(i+1) for i in range(norm_feats.shape[1])])
                y_true = np.array(y_true, dtype=str)
                X = df.to_numpy(dtype=float)
                os.makedirs(svms_dir, exist_ok=True)

                def calc_thrd(crt_id):
                    crt_id = str(crt_id)
                    crt_id_svm_path = os.path.join(svms_dir,crt_id+".model")
                    if (not os.path.exists(crt_id_svm_path)):
                        y = np.empty(y_true.shape, dtype = str)
                        y[y_true==crt_id]='+1'
                        y[y_true!=crt_id]='-1'
                        clf = LinearSVC(penalty='l2', dual=False, tol=toler, C=regul, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=123)
                        clf.fit(X,y)
                        svm_weights = clf.coef_
                        svm_bias = clf.intercept_
                        out_weights = ""
                        for it in range(0, svm_weights.size):
                            out_weights = out_weights+" "+str(svm_weights.item(it))
                        out_weights = out_weights.lstrip()
                        out_bias = str(svm_bias.item(0))
                        with open(crt_id_svm_path,"w") as f_svm:
                            f_svm.write(out_weights+"\n") 
                            f_svm.write(out_bias+"\n")
                print("Training state",curri, "of", S)
                with Pool() as p:
                    p.map(calc_thrd, list(range(min_pos,max_pos)))

