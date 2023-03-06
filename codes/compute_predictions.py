import sys
import os
import numpy as np
from sklearn.preprocessing import Normalizer
from multiprocessing import Pool
from configparser import ConfigParser

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
pred_root = cp['pred_root']

datasets_mean_std_file_path = cp['mean_std']
last = sys.argv[2]
images_list_dir = os.path.join(list_root, normalization_dataset_name)

destination_dir = os.path.join(feat_root, normalization_dataset_name, "seed"+str(random_seed),"b"+str(first_batch_size))
destination_dir_last = os.path.join(feat_root, normalization_dataset_name, "seed"+str(random_seed),"b"+str(first_batch_size),last)
classifiers_dir = os.path.join(classifiers_root, normalization_dataset_name, "seed"+str(random_seed),"b"+str(first_batch_size))
classifiers_dir_last = os.path.join(classifiers_root, normalization_dataset_name, "seed"+str(random_seed),"b"+str(first_batch_size),last)
pred_dir = os.path.join(pred_root, normalization_dataset_name, "seed"+str(random_seed),"b"+str(first_batch_size),last)
print('normalization dataset name = ' + str(normalization_dataset_name))

dataset_name = normalization_dataset_name

state_size = P
nb_classes = P*S


lastint = 4
if last=='half':
   lastint=2
elif last=='all':
    lastint=1

print("Root path:", destination_dir, destination_dir_last)
print("Classifiers path:", classifiers_dir)
print("Number of classes:", nb_classes)
print("Number of states:", S)
print("Number of classes per state:", state_size)
print("dataset name:", normalization_dataset_name)



""" list of arguments for the script """


os.makedirs(pred_dir, exist_ok=True)
def compute_feature(i):
   corresponding_batch = i//(nb_classes//S)+1
   for batchs in range(corresponding_batch, S+1):
      os.makedirs(os.path.join(pred_dir,"batch"+str(batchs)),exist_ok=True)
      list_states = list(range(1,batchs+1))
      for b in range(len(list_states)):
         provenance_batch = list_states[b]
         model_dir = classifiers_dir
         features_dir = destination_dir
         if provenance_batch > 1:
            model_dir = classifiers_dir_last
            features_dir = destination_dir_last
         pred_file = os.path.join(pred_dir,"batch"+str(batchs),str(i)+'batch'+str(provenance_batch))
         if not os.path.exists(pred_file):
            first_class = P*(provenance_batch-1)
            try:
               last_class = P*(list_states[b+1]-1)
            except:
               last_class = P*batchs
            if provenance_batch == 1:
               last_class = P*batchs
            f_list_syn = list(range(first_class,last_class))
            weights_list = []  
            biases_list = []
            for syn in f_list_syn:
               line_cnt = 0 # counter to get the weights and bias lines
               target_model = os.path.join(model_dir,"batch"+str(provenance_batch),str(syn)+".model")
               f_model = open(target_model)
               for line in f_model:
                  line = line.rstrip()
                  if line_cnt == 0:
                     parts = line.split(" ")
                     parts_float = [] # tmp list to store the weights
                     for pp in parts:
                        parts_float.append(float(pp))
                     weights_list.append(parts_float)
                  elif line_cnt == 1:
                     biases_list.append(float(line))
                  line_cnt = line_cnt + 1
               f_model.close()
            test_feats_path = os.path.join(features_dir,f"batch{provenance_batch}","test")
            test_feats = os.path.join(test_feats_path,str(i))
            np_feat = np.loadtxt(test_feats, dtype=float)
            np_feat = np_feat.tolist()
            with open(pred_file, "w") as f_pred:
               for crt_feat in np_feat:
                  crt_feat = Normalizer().fit_transform([crt_feat])[0]
                  pred_dict = []
                  for cls_cnt in range(len(weights_list)):
                     cls_score = np.dot(crt_feat, weights_list[cls_cnt]) + biases_list[cls_cnt]
                     pred_dict.append(str(format(-cls_score, '.10f')))
                  pred_line = ' '.join(pred_dict)
                  f_pred.write(pred_line+"\n")
         else:
            print("exists predictions file:",pred_file)
with Pool() as p:
   p.map(compute_feature, range(nb_classes))