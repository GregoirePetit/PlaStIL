import sys
import os
import numpy as np
import csv
from sklearn.preprocessing import Normalizer
from multiprocessing import Pool
from configparser import ConfigParser


if len(sys.argv) != 4:
    print('Arguments: config, last, method')
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
regul = float(cp["regul"])
toler = float(cp["toler"])
feat_root = cp["feat_root"]
classifiers_root = cp["classifiers_root"]
list_root = cp["list_root"]
model_root = cp["model_root"]
random_seed = int(cp["random_seed"])
num_workers = int(cp['num_workers'])
pred_root = cp['pred_root']
logs_root = cp['logs_root']
datasets_mean_std_file_path = cp['mean_std']

last = sys.argv[2]
method = sys.argv[3]

images_list_dir = os.path.join(list_root, normalization_dataset_name)
destination_dir = os.path.join(feat_root, normalization_dataset_name, "seed"+str(random_seed),"b"+str(first_batch_size))
destination_dir_last = os.path.join(feat_root, normalization_dataset_name, "seed"+str(random_seed),"b"+str(first_batch_size),last)
classifiers_dir = os.path.join(classifiers_root, normalization_dataset_name, "seed"+str(random_seed),"b"+str(first_batch_size))
classifiers_dir_last = os.path.join(classifiers_root, normalization_dataset_name, "seed"+str(random_seed),"b"+str(first_batch_size),last)
pred_dir = os.path.join(pred_root, normalization_dataset_name, "seed"+str(random_seed),"b"+str(first_batch_size),last)
logs_dir = os.path.join(logs_root, normalization_dataset_name, "seed"+str(random_seed),"b"+str(first_batch_size),last)
if method=='deesil':
    logs_dir = os.path.join(logs_root, normalization_dataset_name, "seed"+str(random_seed),"b"+str(first_batch_size))

#print('normalization dataset name = ' + str(normalization_dataset_name))

dataset_name = normalization_dataset_name

state_size = P
batch_size = P
nb_classes = P*S
s= S

lastint = 4
if last=='half':
   lastint=2
elif last=='all':
    lastint=1

""" list of arguments for the script """

def get_top_indices(lst, qnt=5):
    b = sorted(lst)[-qnt:]
    c = [i for i , x in enumerate(lst) if x in b]
    return c
def flatten(t):
    return [item for sublist in t for item in sublist]
def get_list_states(nb_batch, method):
    if method=='deesil':
        return [1]
    elif method=='plastil':
        return [1]+list(range(max(2,nb_batch-lastint+1),nb_batch+1))

def compute_score(nb_batch):
    list_states = get_list_states(nb_batch, method)
    path_pred = os.path.join(pred_dir,"batch"+str(nb_batch))
    y_pred = []
    y_true = []
    score_top5 = []
    for c in range(batch_size*nb_batch):
        with open(os.path.join(path_pred,str(c)+"batch1"), newline='') as f:
            reader = csv.reader(f, delimiter=' ')
            res = np.asarray([elt for elt in reader])
        for state in list_states[1:]:
            with open(os.path.join(path_pred,str(c)+"batch"+str(state)), newline='') as f:
                reader = csv.reader(f, delimiter=' ')
                min_class = (state-1)*batch_size
                max_class = state*batch_size
                res[:,min_class:max_class] = np.asarray([elt for elt in reader])
        scores = [[float(elt[i]) for i in range(batch_size*nb_batch)] for elt in res]
        to_append_top5 = [get_top_indices(elt,1) for elt in scores]
        to_append = [elt[0] for elt in to_append_top5]
        to_append_top5 = [get_top_indices(elt,5) for elt in scores]
        y_pred.append(to_append)
        score_top5.append([c in to_append_top5[i] for i in range(len(to_append))])
        y_true.append([c for _ in to_append])
    y_pred = np.asarray(flatten(y_pred))
    y_pred_top5 = flatten(score_top5)
    y_true = np.asarray(flatten(y_true))
    top1_accuracy = np.mean(y_pred == y_true)
    top5_accuracy = np.mean(y_pred_top5)
    top1_curr_accuracy = np.mean(y_pred[y_true>=batch_size*(nb_batch-1)] == y_true[y_true>=batch_size*(nb_batch-1)])
    if nb_batch>1:
        top1_past_accuracy = np.mean(y_pred[y_true<batch_size*(nb_batch-1)] == y_true[y_true<batch_size*(nb_batch-1)])
    else:
        top1_past_accuracy = top1_curr_accuracy
    return((nb_batch,[top1_accuracy, top5_accuracy, top1_past_accuracy, top1_curr_accuracy]))


log_file = os.path.join(logs_dir, method + ".log")
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
if not os.path.exists(log_file):
    print(f"Computing scores in {logs_dir}")
    with Pool() as p:
        resultats = dict(p.map(compute_score, range(1,s+1)))
    top1=[]
    top5=[]
    top1_past=[]
    top1_current=[]
    with open(log_file, 'w') as f:
        f.write(f'====== dataset = {dataset_name} S ={S} P = {P} last = {last} method = {method} ====== \n')
        print('======',f'dataset = {dataset_name}',f'S ={S}',f'P = {P}',f"last = {last}",f"method = {method}",'======')
        for batch_number in range(1,s+1):
            f.write(f'batch {batch_number}, top1 = {resultats[batch_number][0]:.3f}, top5 = {resultats[batch_number][1]:.3f} , top1_past = {resultats[batch_number][2]:.3f}, top1_current = {resultats[batch_number][3]:.3f} \n')
            top1.append(resultats[batch_number][0])
            top5.append(resultats[batch_number][1])
            top1_past.append(resultats[batch_number][2])
            top1_current.append(resultats[batch_number][3])
        f.write('================================================= \n')
        f.write(f'top1_past = {top1_past}, top1_current = {top1_current} \n')
        f.write('===================  TOTAL  ===================== \n')
        print(f'top1 = {sum(top1[1:])/len(top1[1:]):.3f}, top5 = {sum(top5[1:])/len(top5[1:]):.3f}')
        f.write(f'top1 = {sum(top1[1:])/len(top1[1:]):.3f}, top5 = {sum(top5[1:])/len(top5[1:]):.3f} \n')
else:
    print(f'already computed in {log_file}')