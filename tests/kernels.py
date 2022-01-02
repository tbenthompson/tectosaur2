# flake8: noqa
from tectosaur2.elastic2d import ElasticA, ElasticH, ElasticT, ElasticU
from tectosaur2.laplace2d import (
    AdjointDoubleLayer,
    DoubleLayer,
    Hypersingular,
    SingleLayer,
    double_layer,
    hypersingular,
)

kernel_types = [
    SingleLayer,
    DoubleLayer,
    AdjointDoubleLayer,
    Hypersingular,
    ElasticU,
    ElasticT,
    ElasticA,
    ElasticH,
]
