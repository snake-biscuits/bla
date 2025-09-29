# https://en.wikipedia.org/wiki/Matrix_(mathematics)
# https://en.wikipedia.org/wiki/Invertible_matrix
from __future__ import annotations
from typing import Iterable, List


# TODO: generic MxN matrix class
# TODO: vector.vec3 methods
# -- vec3 -> Mat3x1 * Mat3x3 -> vec3
# -- vec3 -> Mat4x1 * Mat
# TODO: numpy


RawMatrix = List[List[float]]


class Matrix:
    raw: RawMatrix
    num_rows: int
    num_columns: int

    def __init__(self, raw: RawMatrix):
        self.raw = raw
        # TODO: validate raw
        self.num_rows = len(raw)
        self.num_columns = len(raw[0])

    def index(self, row: int, column: int) -> float:
        return self.raw[row][column]

    # TODO: __hash__
    # TODO: __eq__

    def __mul__(self, other: Matrix) -> Matrix:
        if isinstance(other, (int, float)):
            # scalar multiplication
            out = [[
                self.index(row, col) * other
                for row in range(self.num_rows)]
                for col in range(self.num_columns)]
            return Matrix(out)
        elif isinstance(other, Matrix):
            # matrix multiplication
            assert self.n == other.m
            out = [[sum(
                self.index(i, n) * other.index(n, j)
                for n in range(self.num_columns))
                for j in range(other.num_columns)]
                for i in range(self.num_rows)]
            return Matrix(out)

    def inverse(self) -> Matrix:
        if self.m == 3 and self.n == 3:
            return self.inverse_3x3()
        else:
            raise NotImplementedError()

    def inverse_3x3(self) -> Matrix:
        assert self.m == 3 and self.n == 3
        a, b, c = (self.index(0, i) for i in range(3))
        d, e, f = (self.index(1, i) for i in range(3))
        g, h, i = (self.index(2, i) for i in range(3))
        A, D, G = +(e*i - f*h), -(b*i - c*h),  (b*f - c*e)
        B, E, H = -(d*i - f*g),  (a*i - c*g), -(a*f - c*d)
        C, F, I = +(d*h - e*g), -(a*h - b*g),  (a*e - b*d)  # noqa E741
        determinant = a * A + b * B + c * C
        return Matrix([
          [A, D, G],
          [B, E, H],
          [C, F, I]]) * determinant

    def transpose(self) -> Matrix:
        return Matrix([[
            self.index(j, i)
            for j in range(self.num_columns)]
            for i in range(self.num_rows)])

    @classmethod
    def identity(cls, size: int) -> Matrix:
        return cls([[
            1 if i == j else 0
            for i in range(size)]
            for j in range(size)])

    @classmethod
    def from_iterable(cls, iterable: Iterable):
        return cls([[x] for x in iterable])


class Mat4x4:
    array: List[List[float]]

    def __init__(self, array=None, cells=dict()):
        if array is None:  # start as an identity matrix
            self.array = [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]
        else:
            self.array = array
        for (i, j), value in cells.items():
            self[i][j] = value

    def __repr__(self) -> str:
        # return "\n".join([
        #     " ".join(["[", *[f"{self[i][j]}" for j in range(4)],"]"])
        #     for i in range(4)])
        return "".join([
            "Mat4x4([\n",
            ",\n".join([
                " " * 4 + str(self.array[i])
                for i in range(4)]),
            "])"])

    def __getitem__(self, index):
        return self.array[index]

    def __eq__(self, other) -> bool:
        if not isinstance(other, Mat4x4):
            return False
        return all(
            self[i][j] == other[i][j]
            for i in range(4)
            for j in range(4))

    def __add__(self, other: Mat4x4) -> Mat4x4:
        if not isinstance(other, Mat4x4):
            type_ = other.__class__.__name__
            raise NotImplementedError(f"cannot add Mat4x4 with '{type_}'")
        return self.do(lambda i, j: self[i][j] + other[i][j])

    def __mul__(self, other: Mat4x4) -> Mat4x4:
        if isinstance(other, (int, float)):
            return self.do(lambda i, j: self[i][j] * other)
        elif isinstance(other, Mat4x4):
            return self.do(lambda i, j: sum(
                self[i][x] * other[x][j]
                for x in range(4)))
        else:
            type_ = other.__class__.__name__
            raise NotImplementedError(f"cannot multiply Mat4x4 by '{type_}'")

    def do(self, func) -> Mat4x4:
        """powerhouse of the cell"""
        out = Mat4x4()
        for i in range(4):
            for j in range(4):
                out.array[i][j] = func(i, j)
        return out

    def is_valid(self) -> bool:
        return all([
            len(self.array) == 4,
            all(
                len(row) == 4
                for row in self.array),
            all(
                isinstance(cell, (int, float))
                for row in self.array
                for cell in row)])

    def transpose(self) -> Mat4x4:
        return self.do(lambda i, j: self.array[j][i])
