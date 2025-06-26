import fms


def test_version_is_defined():
    assert fms.__version__ is not None


def test_version_attribute_does_not_exist():
    assert not hasattr(fms, "version")
