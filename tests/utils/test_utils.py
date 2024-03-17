import pytest
from fms.utils import smallest_power_greater_than


@pytest.mark.parametrize("greater_than_num,expected", [(1, 2), (5, 8), (16, 32)])
def test_smallest_power_greater_than(greater_than_num, expected):
    actual = smallest_power_greater_than(greater_than_num)
    assert actual == expected
