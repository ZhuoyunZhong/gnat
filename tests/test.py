import os
import random
import numpy as np
import math
import time

import gnat


# Test using SE3 points
# Define a simple distance function for SE3 points
def se3_distance(p1, p2, w=1.0):
    d_position = math.hypot(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
    d_rotation = 1 - abs(
        p1[3] * p2[3] + p1[4] * p2[4] + p1[5] * p2[5] + p1[6] * p2[6]
    )
    return d_position + w * d_rotation


# Rotation sampling
def sample_rotation():
    quat = np.random.uniform(-1, 1, 4)
    quat /= np.linalg.norm(quat)
    return quat


def test_gnat(data_points, query_point, k, radius):
    # Initialize GNAT with the custom distance function
    nn = gnat.NearestNeighborsGNAT()
    nn.set_distance_function(se3_distance)

    # Build data structure
    start_time = time.time()

    nn.add_list(data_points)

    build_time = time.time() - start_time
    print(f"GNAT build time: {build_time:.6f} seconds")

    # Test serialization and deserialization
    nn.save(os.path.dirname(os.path.abspath(__file__)) + "/gnat.dat")
    nn = gnat.NearestNeighborsGNAT()
    nn.set_distance_function(se3_distance)

    print("Loading GNAT from file")
    start_time = time.time()

    nn.load(os.path.dirname(os.path.abspath(__file__)) + "/gnat.dat")

    load_time = time.time() - start_time
    print(f"GNAT load time: {load_time:.6f} seconds")

    # Perform k-nearest neighbors search with GNAT
    start_time = time.time()

    indices, nearest_k = nn.nearest_k(query_point, k)
    nearest_k = np.array(nearest_k)

    k_search_time = time.time() - start_time

    dists = [se3_distance(query_point, data_points[i]) for i in indices]
    print(
        f"GNAT {k}-nearest neighbors search time: {k_search_time:.6f} seconds"
    )
    for i in range(k):
        print(f"Dist: {dists[i]:.4f}- {indices[i]}: {data_points[indices[i]]}")

    # Perform range search with GNAT
    start_time = time.time()

    indices, nearest_r = nn.nearest_r(query_point, radius)
    nearest_r = np.array(nearest_r)

    range_search_time = time.time() - start_time
    print(f"GNAT range search time: {range_search_time:.6f} seconds")
    for i in range(len(nearest_r)):
        print(f"{indices[i]}: {nearest_r[i]}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    np.set_printoptions(suppress=True, precision=3)

    # Create a set of random 2D points
    data_points = np.array(
        [
            [*np.random.uniform(-1, 1, 3), *sample_rotation()]
            for _ in range(10000)
        ]
    )

    # Define the query point
    query_point = np.array([0, 0, 0, 0, 0, 0, 1])
    k = 5
    radius = 0.5

    test_gnat(data_points, query_point, k, radius)
