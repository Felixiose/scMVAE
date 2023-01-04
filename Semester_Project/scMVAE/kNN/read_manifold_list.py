import os
import numpy as np
import pickle

# from ...scMVAE.models import FeedForwardVAE
import argparse
# from ...scMVAE import utils
# from ...utils import str2bool
# from ...data.utils import create_dataset
# from ...scMVAE.components.component import *
#from scipy.spatial import distance
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score

# from ..kNN.distances import euclidean_distance, spherical_distance, spherical_projected_gyro_distance, lorentz_distance, poincare_distance


parser = argparse.ArgumentParser()

parser.add_argument("--file", type=str,  help="Path to a file with labels or batch effects")

args = parser.parse_args()

with open(args.file, 'rb') as f:
    data = pickle.load(f)

for i in data:
    x,y,z =i
 
    print(f'{args.file}\t{x}\t{y}\t{z}')
