from typing import Callable

import equinox as eqx
from jax import nn as jnn
from jaxtyping import Array, Float, Key

from .model import AbstractModel


class MLPModel(
    AbstractModel[[Float[Array, " in_size"]], Float[Array, " out_size"]], strict=True
):
    """Wrapper around eqx.nn.MLP."""

    mlp: eqx.nn.MLP

    in_size: int
    out_size: int

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        activation: Callable[
            [Float[Array, " width"]], Float[Array, " width"]
        ] = jnn.relu,
        final_activation: Callable[
            [Float[Array, " width"]], Float[Array, " width"]
        ] = lambda x: x,
        *,
        key: Key,
    ) -> None:
        self.in_size = in_size
        self.out_size = out_size

        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            final_activation=final_activation,
            key=key,
        )

    def __call__(self, x: Float[Array, " in_size"]) -> Float[Array, " out_size"]:
        return self.mlp(x)
