import random
import numpy as np
import math
import time

import gnat


# Test using SE3 points
# Define a simple distance function for SE3 points
def se3_distance(p1, p2, w=1.0):
    d_position = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    d_rotation = 1 - np.abs(np.dot(p1[3:7], p2[3:7]))
    return d_position + w * d_rotation


# Rotation sampling
def sample_rotation():
    quat = np.random.uniform(-1, 1, 4)
    quat /= np.linalg.norm(quat)
    return quat


def test_gnat():
    # Create a set of random 2D points
    data_points = np.array(
        [
            (
                random.uniform(0, 100),
                random.uniform(0, 100),
                random.uniform(0, 100),
                *sample_rotation(),
            )
            for _ in range(10000)
        ]
    )

    # Define the query point
    query_point = np.array([50, 50, 50, 0, 0, 0, 1])
    k = 5
    radius = 2

    # Initialize OMPL GNAT with the custom distance function
    nn = gnat.NearestNeighborsGNAT()
    nn.set_distance_function(se3_distance)

    # Build data structure
    start_time = time.time()

    nn.add_list(data_points)

    gnat_build_time = time.time() - start_time
    print(f"OMPL GNAT build time: {gnat_build_time:.6f} seconds")

    # Perform nearest neighbor search with OMPL GNAT
    start_time = time.time()

    nearest = nn.nearest(query_point)
    nearest = np.array(nearest)

    gnat_search_time = time.time() - start_time
    print(
        f"OMPL GNAT nearest neighbor search time: {gnat_search_time:.6f} seconds"
    )
    print(f"Nearest neighbor found by OMPL GNAT: \n{nearest}")

    # Perform k-nearest neighbors search with OMPL GNAT
    start_time = time.time()

    nearest_k = nn.nearest_k(query_point, k)
    nearest_k = np.array(nearest_k)

    gnat_k_search_time = time.time() - start_time
    print(
        f"OMPL GNAT {k}-nearest neighbors search time: {gnat_k_search_time:.6f} seconds"
    )
    for i in range(k):
        print(f"{nearest_k[i]}")

    # Perform range search with OMPL GNAT
    start_time = time.time()

    nearest_r = nn.nearest_r(query_point, radius)
    nearest_r = np.array(nearest_r)

    ompl_gnat_range_search_time = time.time() - start_time
    print(
        f"OMPL GNAT range search time: {ompl_gnat_range_search_time:.6f} seconds"
    )
    for i in range(len(nearest_r)):
        print(f"{nearest_r[i]}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    np.set_printoptions(suppress=True, precision=3)

    test_gnat()
