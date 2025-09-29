from bla.matrix import Matrix


def test_matrix_multiplication():
    A = Matrix([
        [2, 3, 4],
        [1, 0, 0]])

    B = Matrix([
        [0, 1000],
        [1, 100],
        [0, 10]])

    C = Matrix([
        [3, 2340],
        [0, 1000]])

    assert A * B == C


def test_scalar_multiplication():
    M = Matrix([
        [1, 2, 3],
        [4, 5, 6]])

    M2 = Matrix([
        [2,  4,  6],
        [8, 10, 12]])

    assert M * 2 == M2
