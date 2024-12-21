import importlib.metadata


def test_version():
    assert importlib.metadata.version("hdfset").count(".") == 2
