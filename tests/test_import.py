import extractor


def test_import():
    assert hasattr(extractor, "__version__")
    assert hasattr(extractor, "__description__")
