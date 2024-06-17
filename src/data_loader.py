import numpy as np
import os

def load_true_coordinates(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    coordinates = [tuple(map(float, line.split())) for line in lines]
    return np.array(coordinates)

# def load_vo_coordinates(file_path, scale):
#     with open(file_path, 'r') as f:
#         lines = f.readlines()
#     coordinates = [tuple(map(float, line.split())) for line in lines]
#     coordinates = np.array(coordinates) * scale
#     return coordinates