# PlaStIL
This is the official code for [PlaStIL](https://gregoirepetit.github.io/projects/PlaStIL) (CoLLAs2023): Plastic and Stable Exemplar-Free Class-Incremental Learning

<p align="center">
<img src="https://gregoirepetit.github.io/images/PlaStIL_thumbnail.png" />
</p>

## Abstract

Plasticity and stability are needed in class-incremental learning in order to learn from new data while preserving past knowledge. Due to catastrophic forgetting, finding a compromise between these two properties is particularly challenging when no memory buffer is available. Mainstream methods need to store two deep models since they integrate new classes using fine-tuning with knowledge distillation from the previous incremental state. We propose a method which has similar number of parameters but distributes them differently in order to find a better balance between plasticity and stability. Following an approach already deployed by transfer-based incremental methods, we freeze the feature extractor after the initial state. Classes in the oldest incremental states are trained with this frozen extractor to ensure stability. Recent classes are predicted using partially fine-tuned models in order to introduce plasticity. Our proposed plasticity layer can be incorporated to any transfer-based method designed for exemplar-free incremental learning, and we apply it to two such methods. Evaluation is done with three large-scale datasets. Results show that performance gains are obtained in all tested configurations compared to existing methods. 

## Results

For the following results *K* denominates the number of states. The baselines results are recomputed using the original configurations of the methods.

### ILSVRC
### Landmarks
### iNaturalist



## Installation

### Environment

To install the required packages, please run the following command (conda is required), using [plastil.yml](plastil.yml) file:

```bash
conda env create -f plastil.yml
```

If the installation fails, please try to install the packages manually with the following command:

```bash
conda create -n plastil python=3.7
conda activate plastil
conda install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
pip install typing-extensions --upgrade
conda install pandas
pip install -U scikit-learn scipy matplotlib
```

### Dependencies

The code depends on the repository [utilsCIL](https://github.com/GregoirePetit/utilsCIL) which contains the code for the datasets and the incremental learning process. Please clone the repository on your home ([PlaStIL code](https://github.com/GregoirePetit/PlaStIL/blob/main/codes/scratch.py#L19) will find it) or add it to your PYTHONPATH:

```bash
git clone git@github.com:GregoirePetit/utilsCIL.git
```

## Usage

The chosen exemple is given for the ILSVRC dataset, for equal number of classes per state (100 classes per state). The code can be adapted to other datasets and other number of classes per state.

### Configuration

Using the [configs/ilsvrc_b100_seed-1.cf](configs/ilsvrc_b100_seed-1.cf) file, you can prepare your experiment. You can change the following parameters:
- `nb_classes`: the total number of classes in the dataset (1000 for ILSVRC, Landmarks and iNaturalist)
- `dataset`: the name of the dataset (ilsvrc for ILSVRC, google_landmarks for Landmarks or inat for iNaturalist)
- `first_batch_size`: the number of classes in the first state, and the number of classes in the other states (in our experiments, we use 50, 100 or 200 classes per state)
- `random_seed`: the random seed used to split the dataset in states, -1 for no random seed
- `num_workers`: the number of workers used to load the data
- `regul`: the regularization parameter used for the linearSVC of the PlaStIL classifiers
- `toler`: the tolerance parameter used for the linearSVC of the PlaStIL classifiers
- `epochs`: the number of epochs used to train the PlaStIL first model
- `list_root`: the path to the list of images of the dataset
- `model_root`: the path to the models
- `feat_root`: the path to the features
- `classifiers_root`: the path to the classifiers
- `pred_root`: the path to the predictions
- `logs_root`: the path to the logs
- `mean_std`: the path to the mean and std of the dataset
- `batch_size`: the batch size used to train the PlaStIL first model
- `lr`: the learning rate used to train the PlaStIL first model
- `momentum`: the momentum used to train the PlaStIL first model
- `weight_decay`: the weight decay used to train the PlaStIL first model
- `lrd`: the learning rate decay used to train the PlaStIL first model

### Experiments

Once the configuration file is ready, you can run the following command to launch the experiment:

#### Train the first model:
```bash
python codes/scratch.py configs/ilsvrc_b100_seed-1.cf
```

#### Train the incremental models according to the PlaStIL method:
```bash
python codes/ft.py configs/ilsvrc_b100_seed-1.cf {last,half,all} {2, ..., S}
```
for example to fine-tune PlaStIL_all:
```bash
python codes/ft.py configs/ilsvrc_b100_seed-1.cf all 2
python codes/ft.py configs/ilsvrc_b100_seed-1.cf all 3
...
python codes/ft.py configs/ilsvrc_b100_seed-1.cf all 10
```
#### Extract the features of the train and test sets:
```bash
python codes/features_extraction.py configs/ilsvrc_b100_seed-1.cf {last,half,all} {1, ..., S}
```
for example to fine-tune PlaStIL_all:
```bash
python codes/features_extraction.py configs/ilsvrc_b100_seed-1.cf all 1
python codes/features_extraction.py configs/ilsvrc_b100_seed-1.cf all 2
...
python codes/features_extraction.py configs/ilsvrc_b100_seed-1.cf all 10
```

#### Train the classifiers:
```bash
python codes/train_classifiers.py configs/ilsvrc_b100_seed-1.cf {last,half,all}
```

#### Compute the predictions on the test set:
```bash
python codes/compute_predictions.py configs/ilsvrc_b100_seed-1.cf {last,half,all}
```

#### Compute the accuracy on the test set:
```bash
python codes/eval.py configs/ilsvrc_b100_seed-1.cf {last,half,all} {deesil,plastil}
```

Logs will be saved in the `logs_root` folder.
