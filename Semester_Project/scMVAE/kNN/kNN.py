import torch
from config import * #TODO FIX it and use it how you want it in the exaple file
from scMVAE.models import FeedForwardVAE
from mvae import utils
from data.scRNADataset import scRNADataset

import json
import os

from scMVAE.components.component import *
import math

import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

from kNN.distances import euclidean_distance, spherical_distance, spherical_projected_gyro_distance, lorentz_distance, poincare_distance


#FIXME: are the distances above already squared??


COMPONENTS = utils.parse_components(MODEL, FIXED_CURVATURE)


def load_model():

    config_file = "./data/adipose/adipose.json"
    configs = json.load(open(config_file, "r"))

    dataset = scRNADataset(batch_size=BATCH_SIZE,
                    data_folder=os.path.dirname(configs['data_file']),
                    data_file=configs['data_file'],
                    label_file=configs['label_file'],
                    batch_files=configs['batch_files'],
                    doubles=DOUBLES)

    model = FeedForwardVAE(h_dim=H_DIM,
                        components=COMPONENTS,
                        dataset=dataset,
                        scalar_parametrization=SCALAR_PARAMETRIZATION)

    return model, dataset
    

def create_loaders(model, dataset):
    
    directory = os.scandir("./chkpt/")
    for file in directory:
        if file.name[0].isdigit():
            CHKPT = "./chkpt/" + file.name
    
    model.load_state_dict(torch.load(CHKPT, map_location=DEVICE))
    print("Loaded model: FeedForwardVAE at epoch", EPOCHS, "from", CHKPT)

    train_loader, test_loader = dataset.create_loaders(BATCH_SIZE)
    
    return train_loader, test_loader

def create_X_y(model, loader):

    X_list = []
    y_list = []

    for batch_idx, (x_mb, y_mb) in enumerate(loader):
        x_mb = x_mb.to(model.device)
        reparametrized, concat_z, x_mb_ = model(x_mb)
        X_list.append(concat_z)
        y_list.append(y_mb)

    big_X = X_list[0]
    X_list.pop(0)

    big_y = y_list[0]
    y_list.pop(0)

    for tensor in X_list:
        big_X = torch.cat((big_X,tensor))

    for tensor in y_list:
        big_y = torch.cat((big_y,tensor))
    
    return big_X, big_y

def create_manifold_list(model):
    
    manifold_list = []

    for i, component in enumerate(model.components):

        manifold_type = type(component)
        dim = component.true_dim
        curvature = float(component.manifold.curvature)

        manifold = (manifold_type, dim, curvature)
        manifold_list.append(manifold)

    return manifold_list


model, dataset = load_model()
train_loader, test_loader = create_loaders(model, dataset)
X_train, y_train = create_X_y(model, train_loader)
X_test, y_test = create_X_y(model, test_loader)
manifold_list = create_manifold_list(model)

#FIXME: what about UniversalComponent?

def distance(a, b):

    a = torch.from_numpy(a).double()
    b = torch.from_numpy(b).double()

    distance_sqd = 0
    counter = 0

    for manifold_type, dim, curvature in manifold_list:

        if curvature != 0:
            radius = torch.Tensor([1/math.sqrt(abs(curvature))]).double()                                        #CHRIS: Removed (+1)
        else:
            radius = None


        if manifold_type in [EuclideanComponent, ConstantComponent]:

            distance_sqd += euclidean_distance(a[counter:counter+dim], b[counter:counter+dim])**2

        elif manifold_type == SphericalComponent:

            distance_sqd += spherical_distance(a[counter:counter+dim], b[counter:counter+dim], radius)**2

        elif manifold_type == HyperbolicComponent:

            distance_sqd += lorentz_distance(a[counter:counter+dim], b[counter:counter+dim], radius)**2

        elif manifold_type == StereographicallyProjectedSphereComponent:

            distance_sqd += spherical_projected_gyro_distance(a[counter:counter+dim], b[counter:counter+dim], curvature)**2    #FIXME: gyro or not gyro? => CHRIS: I think gyro

        elif manifold_type == PoincareComponent:

            distance_sqd += poincare_distance(a[counter:counter+dim], b[counter:counter+dim], radius)

        else:

            distance_sqd = "ERROR"

        counter += dim

    return math.sqrt(distance_sqd)


def kNN(X_train, X_test, y_train, y_test):

    #train
    parameters = {'n_neighbors':[2, 4, 8]}
    clf = GridSearchCV(KNeighborsClassifier(metric=distance, n_jobs=-1), parameters, cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)

    #evaluate
    y_pred = clf.predict(X_test)
    # print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))


X_train = X_train.detach().numpy().astype(np.float64)
X_test = X_test.detach().numpy().astype(np.float64)
y_train = y_train.detach().numpy().astype(np.float64).ravel()
y_test = y_test.detach().numpy().astype(np.float64).ravel()

kNN(X_train, X_test, y_train, y_test)