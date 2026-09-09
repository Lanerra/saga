"""Numeric admission and producer identity; neither certifies semantic quality."""
import hashlib
import json
from typing import Any

import numpy as np

import config
from config.settings import EffectiveSettings


def embedding_identity(configuration: EffectiveSettings | None = None) -> str:
    settings = configuration if configuration is not None else config.snapshot_settings()
    contract = [settings.EMBEDDING_MODEL, settings.EMBEDDING_API_BASE, settings.EXPECTED_EMBEDDING_DIM, settings.EMBEDDING_DTYPE, settings.EMBEDDING_MAX_INPUT_TOKENS, settings.TIKTOKEN_DEFAULT_ENCODING]
    return hashlib.sha256(json.dumps(contract, separators=(',', ':')).encode()).hexdigest()


def validate_embedding(vector: Any, *, model: str, configuration: EffectiveSettings | None = None) -> np.ndarray:
    settings = configuration if configuration is not None else config.snapshot_settings()
    if not isinstance(model, str) or not model.strip() or model != settings.EMBEDDING_MODEL:
        raise ValueError('Embedding model identity mismatch')
    if isinstance(vector, list):
        if any(type(value) not in (int, float) for value in vector):
            raise ValueError('Embedding values must be finite numbers, not bool or nested values')
    elif not isinstance(vector, np.ndarray):
        raise ValueError('Embedding requires a numeric vector')
    array = np.asarray(vector)
    if array.ndim != 1 or array.shape != (settings.EXPECTED_EMBEDDING_DIM,) or array.dtype.kind not in 'fiu':
        raise ValueError('Embedding rank, dimensions or numeric dtype mismatch')
    if not np.isfinite(array).all():
        raise ValueError('Embedding values must be finite')
    with np.errstate(over='ignore', invalid='ignore'):
        converted = array.astype(settings.EMBEDDING_DTYPE)
    if converted.dtype.kind != 'f' or not np.isfinite(converted).all():
        raise ValueError('Embedding conversion must preserve finite floating values')
    return converted
