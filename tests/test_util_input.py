from src.util_input import read_config


def test_util_input_empty_config():
    var_a, var_b = read_config("tests/config_1.txt")
    assert not bool(var_a)
    assert not bool(var_b)


def test_util_input_global_only():
    var_a, var_b = read_config("tests/config_2.txt")
    assert var_a == [1.0]
    assert not bool(var_b)


def test_util_input_planet_only():
    var_a, var_b = read_config("tests/config_3.txt")
    assert not bool(var_a)
    assert var_b == [["Earth", 0.5, 1.0, 2.0, 0.01, 0.02]]


def test_util_input_global_1_planet():
    var_a, var_b = read_config("tests/config_4.txt")
    assert var_a == [1.0]
    assert var_b == [["Earth", 0.5, 1.0, 2.0, 0.01, 0.02]]


def test_util_input_global_2_planets():
    var_a, var_b = read_config("tests/config_5.txt")
    assert var_a == [1.0]
    assert var_b == [
        ["Earth", 0.5, 1.0, 2.0, 0.01, 0.02],
        ["Mars", 0.4, -5.0, 2.0, 0.04, 0.01],
    ]

def test_util_input_global_type():
    var_a, var_b = read_config("tests/config_6.txt")
    assert isinstance(var_a[0], float)
    assert isinstance(var_a[1], float)
    assert isinstance(var_a[2], int)
    assert var_b == []
