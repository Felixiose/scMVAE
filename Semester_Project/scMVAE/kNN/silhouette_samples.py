import torch
import math
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
import gc
from functools import partial
from scipy.spatial.distance import squareform, cdist
from sklearn.metrics import silhouette_score
import argparse

from ...scMVAE.models import FeedForwardVAE
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

def create_loader_sequential(dataset):
    sequential_loader = dataset.create_sequential_loader(args.batch_size)
    return sequential_loader

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

def read_factorize_data(filepath):
        """
        read a file with a single column, factorize to numerical classes 
        return factorized data as numpy array
        """
        names = pd.read_csv(filepath, header=None)
        na_filter = np.where(names.notnull().values)
        fct = pd.factorize(names[0])[0]+1
        return fct, na_filter

model, dataset = load_model()
sequential_loader = create_loader_sequential(model, dataset)
         
X, _ = create_X_y(model, sequential_loader)
X = X.detach().numpy().astype(np.float64)
labels, na_filter = read_factorize_data(args.batch_file)
label_name = os.path.splitext(os.path.basename(args.batch_file))[0]

if X.shape[0] != labels.shape[0]:
    raise("X_all.shape[0] != y_all.shape[0]")

if na_filter[0].shape[0] < X.shape[0] :
    X = X[na_filter[0],]
    labels = labels[na_filter[0],]

manifold_list = create_manifold_list(model)
print(manifold_list)

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

def make_matrix_indices(n):
    mtx_indices = np.triu_indices(n)
    mtx_i=mtx_indices[0] [ np.where(mtx_indices[0]!=mtx_indices[1]) ]
    mtx_j=mtx_indices[1] [ np.where(mtx_indices[0]!=mtx_indices[1]) ]
    return mtx_i, mtx_j

def make_indices_chunks(n,chunk_size):        
    mtx_i, mtx_j = make_matrix_indices (n)
    vect_dim = mtx_i.shape[0]
    vect_indices = np.array(range(0,vect_dim))
    n_chunks = int(np.floor(vect_dim/chunk_size))
    vect_indices_chunks = np.array_split(vect_indices, n_chunks)
    list_chunks= [((mtx_i[i], mtx_j[i], i)) for i in vect_indices_chunks ]
    return list_chunks
    
def compute_dist(chunk, X):
    i_chunck, j_chunk, k_chunk = chunk
    res = np.zeros(i_chunck.shape)
    for i in np.unique(i_chunck):
        ind_i=np.where(i_chunck==i)
        ind_j=j_chunk[ind_i]
        res[ind_i] = cdist(XA = X[i:i+1,:],
                           XB = X[ind_j,],
                           metric = distance)
    return (res, k_chunk)

def compute_silhuette_samples (X, Y, chunk_size):
    n=int(X.shape[0])
    list_chunks = make_indices_chunks(n,chunk_size)
    kwargs = {'X':X }
    mapfunc = partial(compute_dist, **kwargs)
    pool = Pool()
    res_chunks = pool.map( 
    mapfunc, [chunk for chunk in list_chunks] )
    pool.close()
    dist_vec = np.zeros(int((n*n-n)/2))
    for dist, ind in res_chunks:
        dist_vec [ind] = dist
    score = silhouette_score (squareform(dist_vec), Y, metric="precomputed")
    return score
  
indices = list(range(X.shape[0]))

list_samples = list()
for i in range(args.n):
    sample = np.random.choice(indices, size=args.size, replace=False)
    list_samples.append((X[sample],labels[sample]))

del X
del labels

gc.collect()

res_list=list()
for i in list_samples:
    s = compute_silhuette_samples (X=i[0], Y=i[1], chunk_size=100) 
    res_list.append(s)
    print(s)
    
save_path_silh = os.path.dirname(args.chkpt) + "/" +args.id+"_"+label_name+ "_silhouette_score.tsv"

print("Saving results as\n" +  save_path_silh )
with open(save_path_silh, 'w') as tsv:
        tsv.writelines(        
        [ f"{args.id}\t{args.dataset}\t{args.model}\t{args.fixed_curvature}\t{args.universal}\t{args.seed}\t{label_name}\t{silh}\n" for silh in res_list ])
