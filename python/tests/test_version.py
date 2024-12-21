import importlib.metadata


def test_version():
    assert importlib.metadata.version("hdfset") == "0.1.0"
