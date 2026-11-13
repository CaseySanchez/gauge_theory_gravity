"""Adapters from extensor fields to the array reference implementation (`gtg_csta._array`)."""
from gtg_csta import algebra as al


def array_connection(omega):
    """An extensor connection field as the array implementation's function x4 -> (4, 15) columns Omega(e_mu)."""
    def columns(x4):
        return al.connection_columns(omega(al.point(x4)))
    return columns


def array_path(path):
    """An extensor path (scalar parameter -> position) as an array path s -> x4."""
    def points(s):
        return path(al.mv.scalar(s.reshape(1))).kernel
    return points
