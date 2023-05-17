import os
import sys
import pytest

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.data_preprocess import Preprocessing


@pytest.mark.parametrize(
    ('language', 'text'),
    [()]
)
def test_Preprocessing():
    pass
