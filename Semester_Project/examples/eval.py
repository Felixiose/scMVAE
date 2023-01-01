import argparse
import datetime

import torch

from ..data.utils import create_dataset
from ..scMVAE import utils
from ..scMVAE.models import FeedForwardVAE
from ..scMVAE.stats import EpochStats
from ..utils import str2bool


def main() -> None:
    parser = argparse.ArgumentParser(description="M-VAE runner.")
    parser.add_argument("--device", type=str, default="cuda", help="Whether to use cuda or cpu.")
    parser.add_argument("--data", type=str, default="./data", help="Data directory.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size.")
    parser.add_argument("--model", type=str, default="h2,s2,e2", help="Model latent space description.")
    parser.add_argument("--chkpt", type=str, default="", help="Model latent space description.")
    parser.add_argument("--epoch", type=int, default=0, help="Model latent space description.")
    parser.add_argument("--dataset",
                        type=str,
                        default="adipose", 
                        help="Which dataset to run on. Options: adipose, rgc, celegans")
    parser.add_argument("--h_dim", type=int, default=400, help="Hidden layer dimension.")
    parser.add_argument(
        "--scalar_parametrization",
        type=str2bool,
        default=False,
        help="Use a spheric covariance matrix (single scalar) if true, or elliptic (diagonal covariance matrix) if "
        "false.")
    parser.add_argument("--doubles", type=str2bool, default=True, help="Use float32 or float64. Default float32.")
    parser.add_argument("--likelihood_n",
                        type=int,
                        default=500,
                        help="How many samples to use for LL estimation. Value 0 disables LL estimation.")
    args = parser.parse_args()

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

    dataset = create_dataset(dataset_type = args.dataset, batch_size=args.batch_size, doubles = args.doubles) 
    print("#####")
    cur_time = datetime.datetime.utcnow().isoformat()
    components = utils.parse_components(args.model, False)
    model_name = utils.canonical_name(components)
    print(f"Eval VAE Model: {model_name}; Time: {cur_time}; Dataset: {args.dataset}")
    print("#####", flush=True)

    if args.dataset in ["adipose", "rgc", "celegans", "adipose_gaussian"]:      #TODO -> ANDREI TELL THE NAME OF THE DATASET
        model_cls = FeedForwardVAE  # type: ignore
    else:
        raise ValueError(f"Unknown dataset '{args.dataset}'.")
    model = model_cls(h_dim=args.h_dim,
                      components=components,
                      dataset=dataset,
                      scalar_parametrization=args.scalar_parametrization).to(args.device)
    model.load_state_dict(torch.load(args.chkpt, map_location=args.device))

    print("Loaded model", model_name, "at epoch", args.epoch, "from", args.chkpt)
    _, test_loader = dataset.create_loaders(batch_size=args.batch_size)

    print(f"\tEpoch {args.epoch}:\t", end="")
    model.eval()

    batch_stats = []
    for batch_idx, (x_mb, y_mb) in enumerate(test_loader):
        x_mb = x_mb.to(model.device)
        reparametrized, concat_z, x_mb_ = model(x_mb)
        stats = model.compute_batch_stats(x_mb, x_mb_, reparametrized, likelihood_n=args.likelihood_n, beta=1.)
        batch_stats.append(stats.convert_to_float())

    epoch_stats = EpochStats(batch_stats, length=len(test_loader.dataset))
    epoch_dict = epoch_stats.to_print()
    for i, component in enumerate(model.components):
        name = f"{component.summary_name(i)}/curvature"
        epoch_dict[name] = float(component.manifold.curvature)
    print(epoch_dict, flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    # with torch.autograd.set_detect_anomaly(True):
    main()
