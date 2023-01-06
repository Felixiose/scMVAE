import torch
import math
import os
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from ...scMVAE.models import FeedForwardVAE
import argparse
from ...scMVAE import utils
from ...utils import str2bool
from ...data.utils import create_dataset
from ...scMVAE.components.component import *
from ..kNN.distances import *


parser = argparse.ArgumentParser(description="M-VAE runner.")
parser.add_argument("--id", type=str, default="id", help="A custom run id to keep track of experiments")
parser.add_argument("--device", type=str, default="cuda", help="Whether to use cuda or cpu.")
parser.add_argument("--batch_size", type=int, default=100, help="Batch size.")
parser.add_argument("--chkpt", type=str, default="", help="Model latent space description.")
parser.add_argument("--model", type=str, default="h2,s2,e2", help="Model latent space description.")
parser.add_argument("--universal", type=str2bool, default=False, help="Universal training scheme.")
parser.add_argument("--dataset",
                    type=str,
                    default="adipose", 
                    help="Which dataset to run on. Options: adipose, rgc, celegans")
parser.add_argument("--h_dim", type=int, default=400, help="Hidden layer dimension.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument(
    "--scalar_parametrization",
    type=str2bool,
    default=False,
    help="Use a spheric covariance matrix (single scalar) if true, or elliptic (diagonal covariance matrix) if "
    "false.")
parser.add_argument("--fixed_curvature",
                    type=str2bool,
                    default=True,
                    help="Whether to fix curvatures to (-1, 0, 1).")
parser.add_argument("--doubles", type=str2bool, default=True, help="Use float32 or float64. Default float32.")
args = parser.parse_args()

if args.seed:
    print("Using pre-set random seed:", args.seed)
    utils.set_seeds(args.seed)

if not torch.cuda.is_available():
    args.device = "cpu"
    print("CUDA is not available.")
args.device = torch.device(args.device)
utils.setup_gpu(args.device)
print("Running on:", args.device, flush=True)

if args.doubles:
    torch.set_default_dtype(torch.float64)
else:
    torch.set_default_dtype(torch.float32)

print(args.model)
print(args.fixed_curvature)
COMPONENTS = utils.parse_components(args.model, args.fixed_curvature)

def load_model():
    dataset = create_dataset(dataset_type = args.dataset, batch_size=args.batch_size, doubles = args.doubles) 
    model = FeedForwardVAE(h_dim=args.h_dim,
                        components=COMPONENTS,
                        dataset=dataset,
                        scalar_parametrization=args.scalar_parametrization)
    model.load_state_dict(torch.load(args.chkpt, map_location=args.device))
    print("Loaded model: FeedForwardVAE at epoch", args.epochs, "from", args.chkpt)
    return model, dataset
    
def create_loaders(dataset):
    train_loader, test_loader = dataset.create_loaders(args.batch_size)
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
        dim = component.dim
        curvature = float(component.manifold.curvature)

        if manifold_type == UniversalComponent:
            if curvature > 0:
                manifold_type = StereographicallyProjectedSphereComponent
            elif curvature < 0:
                manifold_type = PoincareComponent
            else:
                manifold_type = EuclideanComponent

        manifold = (manifold_type, dim, curvature)
        manifold_list.append(manifold)

    return manifold_list

model, dataset = load_model()
train_loader, test_loader = create_loaders(model, dataset)
X_train, y_train = create_X_y(model, train_loader)
X_test, y_test = create_X_y(model, test_loader)
manifold_list = create_manifold_list(model)

def distance(a, b):

    a = torch.from_numpy(a).double()
    b = torch.from_numpy(b).double()

    distance_sqd = 0
    counter = 0

    for manifold_type, dim, curvature in manifold_list:

        if curvature != 0:
            radius = torch.Tensor([1/math.sqrt(abs(curvature))]).double()
        else:
            radius = None

        curvature = torch.Tensor([curvature]).double()

        if manifold_type in [EuclideanComponent, ConstantComponent]:
            distance_sqd += euclidean_distance(a[counter:counter+dim], b[counter:counter+dim])**2
        elif manifold_type == SphericalComponent:
            distance_sqd += spherical_distance(a[counter:counter+dim], b[counter:counter+dim], radius)**2
        elif manifold_type == HyperbolicComponent:
            distance_sqd += lorentz_distance(a[counter:counter+dim], b[counter:counter+dim], radius)**2
        elif manifold_type == StereographicallyProjectedSphereComponent:
            distance_sqd += spherical_projected_gyro_distance(a[counter:counter+dim], b[counter:counter+dim], curvature)**2
        elif manifold_type == PoincareComponent:
            distance_sqd += poincare_distance(a[counter:counter+dim], b[counter:counter+dim], radius)**2
        else:
            distance_sqd = "ERROR"

        counter += dim

    return math.sqrt(distance_sqd)

def kNN(X_train, X_test, y_train, y_test):

    # train
    parameters = {'n_neighbors':[2, 4, 8]}
    clf = GridSearchCV(KNeighborsClassifier(metric=distance, n_jobs=-1), parameters, cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)

    # evaluate
    y_pred = clf.predict(X_test)
    return (accuracy_score(y_test, y_pred),clf)

X_train = X_train.detach().numpy().astype(np.float64)
X_test = X_test.detach().numpy().astype(np.float64)
y_train = y_train.detach().numpy().astype(np.float64).ravel()
y_test = y_test.detach().numpy().astype(np.float64).ravel()

n_replicates = 10
size = 1000
indices = list(range(X_train.shape[0]-1))

res_acc = list()
res_classifier = list()

for i in range(n_replicates):
    sample = np.random.choice(indices, size=size, replace=False)
    
    accuracy, classifier = kNN(X_train[sample,], X_test[sample], y_train[sample], y_test[sample])
    res_acc.append(accuracy)
    res_classifier.append(classifier)
    
save_path_acc = os.path.dirname(args.chkpt) + "/" +args.id+ "_knn_sample_accuracy.tsv"
save_path_clf = os.path.dirname(args.chkpt) + "/" +args.id+ "_knn_sample_classifier.pickle"
save_path_manifold = os.path.dirname(args.chkpt) + "/" +args.id+ "_manifold_list.pickle"

print("Saving results in" + os.path.dirname(args.chkpt) )

with open(save_path_acc, 'w') as tsv:
        tsv.writelines(        
        [ f"{args.id}\t{args.dataset}\t{args.model}\t{args.fixed_curvature}\t{args.universal}\t{args.seed}\t{accuracy}\n" for accuracy in res_acc ])

with open(save_path_clf, 'wb') as f:
    pickle.dump(res_classifier, f)

with open(save_path_manifold, 'wb') as f:
    pickle.dump(manifold_list, f)