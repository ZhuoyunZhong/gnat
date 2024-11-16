import random
import numpy as np

import gnat
from test import se3_distance


def test(data_points, query_point):
    # In order to remove data, we need to use the point handler
    # returned by the add_list or add method
    data_handles = []

    nn = gnat.NearestNeighborsGNAT()
    nn.set_distance_function(se3_distance)

    print("Add the first 2 data")
    data_handle1 = nn.add_list(data_points[:2])
    indices, nearest = nn.nearest_k(query_point, 2)
    for i in range(2):
        print(f"Nearest neighbor {i}: {indices[i]}, {nearest[i]}")

    print("Add the last data")
    data_handle2 = nn.add(data_points[2])
    indices, nearest = nn.nearest_k(query_point, 2)
    for i in range(2):
        print(f"Nearest neighbor {i}: {indices[i]}, {nearest[i]}")

    print("Remove the first data")
    data_handles.extend(data_handle1)
    data_handles.append(data_handle2)
    nn.remove(data_handles[0])
    indices, nearest = nn.nearest_k(query_point, 2)
    for i in range(2):
        print(f"Nearest neighbor {i}: {indices[i]}, {nearest[i]}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    np.set_printoptions(suppress=True, precision=3)

    # Create a set of random 2D points
    data_points = np.array(
        [
            [0.0, 1, 0, 0, 0, 0, 1],
            [0.0, 2, 0, 0, 0, 0, 1],
            [0.0, 0, 0, 0, 0, 0, 1],
        ]
    )
    # Define the query point
    query_point = np.array([0.0, 0, 0, 0, 0, 0, 1])

    test(data_points, query_point)
