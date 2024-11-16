import os
import random
import numpy as np
import time
import pickle

from test import se3_distance, sample_rotation, test_gnat
from sklearn.neighbors import BallTree


def test_balltree(data_points, query_point, k, radius):
    # Build data structure
    start_time = time.time()

    nn = BallTree(data_points, metric=se3_distance)

    build_time = time.time() - start_time
    print(f"Balltree build time: {build_time:.6f} seconds")

    # Test serialization and deserialization
    with open(
        os.path.dirname(os.path.abspath(__file__)) + "/balltree.pkl", "wb"
    ) as f:
        pickle.dump(nn, f)

    print("Loading Balltree from file")
    start_time = time.time()

    with open(
        os.path.dirname(os.path.abspath(__file__)) + "/balltree.pkl", "rb"
    ) as f:
        nn = pickle.load(f)

    load_time = time.time() - start_time
    print(f"Balltree load time: {load_time:.6f} seconds")

    # For search
    query_array = np.array([query_point])

    # Perform k-nearest neighbors search
    start_time = time.time()

    indices = nn.query(query_array, k=k, return_distance=False)[0]

    k_search_time = time.time() - start_time

    dists = [se3_distance(query_point, data_points[i]) for i in indices]
    print(
        f"Balltree {k}-nearest neighbors search time: {k_search_time:.6f} seconds"
    )
    for i in range(k):
        print(f"Dist: {dists[i]:.4f}- {indices[i]}: {data_points[indices[i]]}")

    # Perform range search
    start_time = time.time()

    indices = nn.query_radius(
        query_array, radius, return_distance=True, sort_results=True
    )[0][0]

    range_search_time = time.time() - start_time
    print(f"Balltree range search time: {range_search_time:.6f} seconds")
    for i in range(len(indices)):
        print(f"{indices[i]}: {data_points[indices[i]]}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    np.set_printoptions(suppress=True, precision=3)

    # Create a set of random 2D points
    data_points = np.array(
        [
            # [*np.random.uniform(-1, 1, 3), *sample_rotation()]
            [
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                *sample_rotation(),
            ]
            for _ in range(100000)
        ]
    )

    # Define the query point
    query_point = np.array([0, 0, 0, 0, 0, 0, 1])
    k = 5
    radius = 0.2

    test_gnat(data_points, query_point, k, radius)
    print("----------------------------------------")
    test_balltree(data_points, query_point, k, radius)
