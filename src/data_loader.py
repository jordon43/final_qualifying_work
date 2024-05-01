import numpy as np

def load_true_coordinates(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    coordinates = [tuple(map(float, line.split())) for line in lines]
    return np.array(coordinates)
