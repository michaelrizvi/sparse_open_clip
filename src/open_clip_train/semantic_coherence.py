import torch
from tqdm import tqdm
import numpy as np

import itertools

import torch
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import os
import sys
sys.path.append('src') # Add the src directory to the Python path.
import open_clip
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from open_clip_train.distributed import is_master
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast
from open_clip import get_input_dtype
from open_clip_train.params import parse_args


coarse_label = [
'apple', # id 0
'aquarium_fish',
'baby',
'bear',
'beaver',
'bed',
'bee',
'beetle',
'bicycle',
'bottle',
'bowl',
'boy',
'bridge',
'bus',
'butterfly',
'camel',
'can',
'castle',
'caterpillar',
'cattle',
'chair',
'chimpanzee',
'clock',
'cloud',
'cockroach',
'couch',
'crab',
'crocodile',
'cup',
'dinosaur',
'dolphin',
'elephant',
'flatfish',
'forest',
'fox',
'girl',
'hamster',
'house',
'kangaroo',
'computer_keyboard',
'lamp',
'lawn_mower',
'leopard',
'lion',
'lizard',
'lobster',
'man',
'maple_tree',
'motorcycle',
'mountain',
'mouse',
'mushroom',
'oak_tree',
'orange',
'orchid',
'otter',
'palm_tree',
'pear',
'pickup_truck',
'pine_tree',
'plain',
'plate',
'poppy',
'porcupine',
'possum',
'rabbit',
'raccoon',
'ray',
'road',
'rocket',
'rose',
'sea',
'seal',
'shark',
'shrew',
'skunk',
'skyscraper',
'snail',
'snake',
'spider',
'squirrel',
'streetcar',
'sunflower',
'sweet_pepper',
'table',
'tank',
'telephone',
'television',
'tiger',
'tractor',
'train',
'trout',
'tulip',
'turtle',
'wardrobe',
'whale',
'willow_tree',
'wolf',
'woman',
'worm',
]

mapping = ['beaver', 'dolphin', 'otter', 'seal', 'whale'] + \
['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']+ \
['orchid', 'poppy', 'rose', 'sunflower', 'tulip']+ \
['bottle', 'bowl', 'can', 'cup', 'plate']+ \
['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper']+ \
['clock', 'computer_keyboard', 'lamp', 'telephone', 'television']+ \
['bed', 'chair', 'couch', 'table', 'wardrobe']+ \
['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach']+ \
['bear', 'leopard', 'lion', 'tiger', 'wolf']+ \
['bridge', 'castle', 'house', 'road', 'skyscraper']+ \
['cloud', 'forest', 'mountain', 'plain', 'sea']+ \
['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo']+ \
['fox', 'porcupine', 'possum', 'raccoon', 'skunk']+ \
['crab', 'lobster', 'snail', 'spider', 'worm']+ \
['baby', 'boy', 'girl', 'man', 'woman']+ \
['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle']+ \
['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']+ \
['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree']+ \
['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train']+ \
['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']

super_classes = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
['bottle', 'bowl', 'can', 'cup', 'plate'],
['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
['clock', 'computer_keyboard', 'lamp', 'telephone', 'television'],
['bed', 'chair', 'couch', 'table', 'wardrobe'],
['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
['bear', 'leopard', 'lion', 'tiger', 'wolf'],
['bridge', 'castle', 'house', 'road', 'skyscraper'],
['cloud', 'forest', 'mountain', 'plain', 'sea'],
['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
['crab', 'lobster', 'snail', 'spider', 'worm'],
['baby', 'boy', 'girl', 'man', 'woman'],
['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]

super_classes = {c: i for i, s in enumerate(super_classes) for c in s}


def is_super(c1, c2):
    return super_classes[c1] == super_classes[c2]


def compute_semantic_coherence(weights):
    super_c_acc = []
    for N in tqdm(range(1, 50)):
        sorted_weights = np.argsort(weights, 1)[:, -N:]
        ut = np.unique(sorted_weights)

        correct = 0
        total = 0
        for i, u in enumerate(ut):
            c = np.argwhere(sorted_weights == u)[:,0]
            for p1, p2 in itertools.permutations(c, 2):
                c1, c2 = coarse_label[p1], coarse_label[p2]
                correct += is_super(c1, c2)
                total += 1
        super_c_acc.append(correct / total)
    return super_c_acc


def train_cifar_classifier(train_activations, train_labels, val_activations, val_labels):
    # Make classifier and train it
    logreg_clf = make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', solver='saga', max_iter=2, verbose=1))
    logreg_clf.fit(train_activations, train_labels)
     
    # Test classifier on validation set
    logreg_predictions = logreg_clf.predict(val_activations)
    logreg_accuracy = accuracy_score(val_labels, logreg_predictions)

    # Get weight matrix
    cifar_classifier_weights = logreg_clf.named_steps['logisticregression'].coef_

    return cifar_classifier_weights, logreg_accuracy


def load_clip_model(weight_path=None, model_name='ViT-B-32', pretrained='laion400m_e32'):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    if weight_path:
        model.load_state_dict(torch.load(weight_path))
    model.eval()
    return model, preprocess


def get_activation_data(model, preprocess, args, train=False):
    device = torch.device(args.device)
    input_dtype = get_input_dtype(args.precision)
    autocast = get_autocast(args.precision)

    # Load CIFAR-100 dataset
    cifar100 = datasets.CIFAR100(root='./data', train=train, download=True, transform=preprocess)

    dataloader = DataLoader(cifar100, batch_size=256, shuffle=False)

    all_activations = []
    all_labels = []
    for images, labels in tqdm(dataloader):
        with torch.no_grad():
            with autocast():
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                activations = model.encode_image(images)
                all_activations.append(activations.cpu())
                all_labels.append(labels.cpu())
    
    activations = torch.cat(all_activations)
    labels = torch.cat(all_labels)
    
    return activations, labels


if __name__ == "__main__":
    args = None
    weight_path = None
    model, preprocess = load_clip_model(weight_path)
    
    val_acts, val_labels = get_activation_data(model, preprocess, args, train=False)
    print(val_acts.shape, val_labels.shape)
