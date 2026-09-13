"""Local cache contract and installed spaCy evidence without network access."""
from typing import Any

import pytest
import spacy

import core.lightweight_cache as cache
from core.spacy_service import SpacyService

# Retain only this library loader before the missing-model fixture runs.
# Filesystem/network isolation and every other offline fixture remain active.
_INSTALLED_SPACY_LOAD = spacy.load


@pytest.mark.parametrize("size", [-1, 0, True, False, 1.5, "2"])
def test_cache_constructor_rejects_nonpositive_or_noninteger_size(size: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        cache._ServiceCache(maxsize=size)


@pytest.mark.parametrize("size", [-1, 0, True, False, 1.5, "2"])
@pytest.mark.parametrize("existing", [False, True])
def test_cache_registration_rejects_size_without_mutating_prior_cache(monkeypatch: pytest.MonkeyPatch, size: Any, existing: bool) -> None:
    monkeypatch.setattr(cache, "_SERVICE_CACHES", {})
    if existing:
        cache.register_cache_service("r03", maxsize=2)
        cache.set_cached_value("first", "retained", "r03")
    before = dict(cache._SERVICE_CACHES)
    with pytest.raises(ValueError, match="positive integer"):
        cache.register_cache_service("r03", maxsize=size)
    assert cache._SERVICE_CACHES == before
    if existing:
        assert cache.get_cache_metrics("r03") == {"size": 1, "maxsize": 2}
        assert cache.get_cached_value("first", "r03") == "retained"
        cache.set_cached_value("second", "still usable", "r03")
        assert cache.get_cache_size("r03") == 2


def test_cache_resize_enforces_lru_bound_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache, "_SERVICE_CACHES", {})
    cache.register_cache_service("r03", maxsize=3)
    for key in ("first", "second", "third"):
        cache.set_cached_value(key, key, "r03")
    assert cache.get_cached_value("first", "r03") == "first"
    cache.register_cache_service("r03", maxsize=1)
    assert cache.get_cache_metrics("r03") == {"size": 1, "maxsize": 1}
    assert cache.get_cached_value("first", "r03") == "first"
    assert cache.get_cached_value("second", "r03") is None
    assert cache.get_cached_value("third", "r03") is None


def test_installed_statistical_spacy_preserves_prose() -> None:
    model = _INSTALLED_SPACY_LOAD("en_core_web_lg")
    assert "tagger" in model.pipe_names
    assert "parser" in model.pipe_names
    assert "ner" in model.pipe_names
    service = SpacyService()
    service._nlp = model
    prose = "Iven's ledger's clasp wouldn't open; you'd think he'd know.\n\n“Don’t,” she said."
    tokens = [token.text for token in model(prose)]
    assert "'s" in tokens and "'d" in tokens and "n't" in tokens
    assert service.clean_text(prose) == prose
