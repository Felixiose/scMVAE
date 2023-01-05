from typing import Any
import torch
from torch import Tensor
import geoopt.manifolds.poincare.math as pm
from ...scMVAE.ops.common import sqrt, atanh
from ...scMVAE.ops.hyperbolics import acosh, lorentz_product


def _c(radius: Tensor) -> Tensor:
    return 1 / radius**2

def poincare_distance(x: Tensor, y: Tensor, radius: Tensor, **kwargs: Any) -> Tensor:
    return poincare_distance_c(x, y, _c(radius), **kwargs)

def poincare_distance_c(x: Tensor, y: Tensor, c: Tensor, keepdim: bool = True, **kwargs: Any) -> Tensor:
    # res = pm.dist(x, y, c=c, keepdim=keepdim, **kwargs)

    sqrt_c = sqrt(c)
    mob = pm.mobius_add(-x, y, c=c, dim=-1).norm(dim=-1, p=2, keepdim=keepdim)
    arg = sqrt_c * mob
    dist_c = atanh(arg)
    res = dist_c * 2 / sqrt_c
    assert torch.isfinite(res).all()
    return res


def mob_add(x: Tensor, y: Tensor, K: Tensor) -> Tensor:
    return pm.mobius_add(x, y, c=-K)

#Definiton of the spherical projected distance
def spherical_projected_distance(x: Tensor, y: Tensor, K: Tensor, **kwargs: Any) -> Tensor:
    diff = x - y
    normxmy2 = torch.sum(diff * diff, dim=-1, keepdim=True)
    normx2 = torch.sum(x * x, dim=-1, keepdim=True)
    normy2 = torch.sum(y * y, dim=-1, keepdim=True)
    dist = 1. / sqrt(K) * torch.acos(torch.clamp(1 - 2 * K * normxmy2 / ((1 + K * normx2) * (1 + K * normy2)), max=1.0))
    assert torch.isfinite(dist).all()
    return dist

#Definition of the spherical projected gyro distance
def spherical_projected_gyro_distance(x: Tensor, y: Tensor, K: Tensor, **kwargs: Any) -> Tensor:
    sqrt_K = sqrt(K)
    sm = mob_add(-x, y, K)
    normxy = torch.norm(sm, p=2, dim=-1, keepdim=True)
    return 2. / sqrt_K * torch.atan(sqrt_K * normxy)


#Definition of the euclidean distance
def euclidean_distance(x: torch.Tensor, y: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    return 2 * torch.norm(x - y, dim=-1, p=2, keepdim=True)


#Definition of the spherical distances
def spherical_distance(x: torch.Tensor, y: torch.Tensor, radius: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    ndot = torch.sum(x * y, dim=-1, keepdim=True) / radius**2
    acos = torch.acos(torch.clamp(ndot, min=-1., max=1.))
    return radius * acos


def lorentz_distance(x: torch.Tensor, y: torch.Tensor, radius: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    return radius * acosh(-lorentz_product(x, y, **kwargs) / (radius**2))


#Definition of the lorentz product
def lorentz_product(x: Tensor, y: Tensor, keepdim: bool = False, dim: int = -1) -> Tensor:

    m = x * y
    if keepdim:
        ret = torch.sum(m, dim=dim, keepdim=True) - 2 * m[..., 0:1]
    else:
        ret = torch.sum(m, dim=dim, keepdim=False) - 2 * m[..., 0]
    return ret
