from uuid import uuid4

import numpy as np
import onnx
from onnxconverter_common.data_types import FloatTensorType
from onnxconverter_common.topology import convert_topology

from ..coniferest import Coniferest
from ._parse import parse_coniferest
from .proto import get_maximum_opset_supported


def _guess_initial_types(X, model, initial_types):
    if initial_types is not None:
        return initial_types

    if isinstance(X, np.ndarray):
        shape = [None, X.shape[1]]
        initial_types = [("X", FloatTensorType(shape))]

    if len(model.trees):
        shape = [None, model.trees[0].n_features]
        initial_types = [("X", FloatTensorType(shape))]

    return initial_types


def convert(
    model,
    name=None,
    initial_types=None,
    doc_string="",
    target_opset=None,
    targeted_onnx=onnx.__version__,
    custom_conversion_functions=None,
    custom_shape_calculators=None,
):
    """
    Convert coniferest model to ONNX.

    Parameters
    ----------
    model : Coniferest
        Coniferest model to convert.

    name : string
        ONNX model name.

    Returns
    -------
    ONNX model

    Examples
    --------
    >>> from coniferest.aadforest import AADForest
    >>> from coniferest.datasets import single_outlier
    >>> from coniferest.onnx import convert
    >>> from onnxconverter_common.data_types import FloatTensorType
    >>>
    >>> data, _metadata = single_outlier()
    >>> data = data.astype(np.float32)
    >>> model = AADForest().fit(data)
    >>> o = convert(model, initial_types = [('X', FloatTensorType([None, data.shape[1]]))])
    """

    if not isinstance(model, Coniferest):
        raise ValueError("model is not Coniferest object")

    if name is None:
        name = str(uuid4().hex)

    if initial_types is None:
        raise ValueError("Initial types are required. See usage of convert(...) in coniferest.onnx.convert for details")

    target_opset = target_opset if target_opset else get_maximum_opset_supported()

    topology = parse_coniferest(
        model,
        initial_types,
        target_opset,
        custom_conversion_functions,
        custom_shape_calculators,
    )
    topology.compile()

    onnx_model = convert_topology(topology, name, doc_string, target_opset, targeted_onnx)
    return onnx_model


def to_onnx(
    model,
    X=None,
    name=None,
    initial_types=None,
    target_opset=None,
    targeted_onnx=onnx.__version__,
    custom_conversion_functions=None,
    custom_shape_calculators=None,
):
    if name is None:
        name = "ONNX(%s)" % model.__class__.__name__

    initial_types = _guess_initial_types(X, model, initial_types)

    return convert(
        model,
        name=name,
        initial_types=initial_types,
        target_opset=target_opset,
        targeted_onnx=targeted_onnx,
        custom_conversion_functions=custom_conversion_functions,
        custom_shape_calculators=custom_shape_calculators,
    )


def save_onnx_model(model, filename=None):
    """
    Serialize ONNX model and save it to a file.

    Parameters
    ----------
    model
        ONNX model to serialize.

    filename
        filename or file object to save the model.

    Returns
    -------
    Serialized ONNX model as bytes
    """

    content = model.SerializeToString()
    if filename is not None:
        if hasattr(filename, "write"):
            filename.write(content)
        else:
            with open(filename, "wb") as f:
                f.write(content)
    return content
