"""Unit tests for the DataCache caching layer."""

import os
import tempfile
import time

import numpy as np
import pandas as pd
import pytest

from src.data.data_cache import DataCache, TTL_PRICE, TTL_FINANCIALS, TTL_DEFAULT


@pytest.fixture
def cache():
    """Create a DataCache backed by a temporary SQLite file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    c = DataCache(db_path=path)
    yield c
    c.clear()
    os.unlink(path)


class TestBasicOperations:
    def test_set_and_get(self, cache):
        cache.set("key1", {"hello": "world"}, ttl_seconds=3600)
        assert cache.get("key1") == {"hello": "world"}

    def test_miss_returns_none(self, cache):
        assert cache.get("nonexistent_key") is None

    def test_overwrite(self, cache):
        cache.set("key1", "v1", ttl_seconds=3600)
        cache.set("key1", "v2", ttl_seconds=3600)
        assert cache.get("key1") == "v2"

    def test_delete(self, cache):
        cache.set("del_me", "value", ttl_seconds=3600)
        cache.delete("del_me")
        assert cache.get("del_me") is None

    def test_delete_nonexistent_no_error(self, cache):
        cache.delete("never_existed")

    def test_clear(self, cache):
        cache.set("a", 1, ttl_seconds=3600)
        cache.set("b", 2, ttl_seconds=3600)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None


class TestTTL:
    def test_entry_expires(self, cache):
        cache.set("short", "live", ttl_seconds=1)
        assert cache.get("short") == "live"
        time.sleep(1.2)
        assert cache.get("short") is None  # expired

    def test_entry_not_expired_within_ttl(self, cache):
        cache.set("long", "live", ttl_seconds=3600)
        assert cache.get("long") == "live"

    def test_explicit_expire(self, cache):
        cache.set("exp1", "v", ttl_seconds=0)  # already expired
        cache.set("exp2", "v", ttl_seconds=0)
        removed = cache.expire()
        assert removed >= 2
        assert cache.get("exp1") is None


class TestNumpyPandasSerialization:
    def test_numpy_scalars(self, cache):
        cache.set("np", {
            "float": np.float64(3.14159),
            "int": np.int64(42),
            "bool": np.bool_(True),
        }, ttl_seconds=3600)
        result = cache.get("np")
        assert abs(result["float"] - 3.14159) < 0.001
        assert result["int"] == 42
        assert result["bool"] is True

    def test_numpy_array(self, cache):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        cache.set("arr", arr, ttl_seconds=3600)
        result = cache.get("arr")
        assert result == [1.0, 2.0, 3.0, 4.0]

    def test_dataframe_roundtrip(self, cache):
        df = pd.DataFrame(
            {"A": [1, 2, 3], "B": [4.0, 5.0, 6.0]}, index=["x", "y", "z"]
        )
        cache.set("df", df, ttl_seconds=3600)
        result = cache.get("df")
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, df)

    def test_series_roundtrip(self, cache):
        s = pd.Series([10, 20, 30], name="my_series")
        cache.set("series", s, ttl_seconds=3600)
        result = cache.get("series")
        assert isinstance(result, pd.Series)
        pd.testing.assert_series_equal(result, s)

    def test_empty_dataframe(self, cache):
        df = pd.DataFrame()
        cache.set("empty_df", df, ttl_seconds=3600)
        result = cache.get("empty_df")
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestStats:
    def test_stats_after_operations(self, cache):
        cache.clear()
        cache.set("a", 1, ttl_seconds=3600)
        cache.set("b", 2, ttl_seconds=3600)
        cache.get("a")  # hit
        cache.get("c")  # miss
        s = cache.stats()
        assert s["total_entries"] >= 2
        assert s["hits"] >= 1
        assert s["misses"] >= 1
        assert s["db_size_bytes"] > 0

    def test_hit_rate(self, cache):
        cache.clear()
        # Reset internal counters by calling clear
        cache.set("x", 10, ttl_seconds=3600)
        cache.get("x")  # hit
        cache.get("y")  # miss
        s = cache.stats()
        assert 0.0 <= s["approximate_hit_rate"] <= 1.0


class TestTTLConstants:
    def test_ttl_values(self):
        assert TTL_PRICE == 86_400
        assert TTL_FINANCIALS == 604_800
        assert TTL_DEFAULT == 86_400
