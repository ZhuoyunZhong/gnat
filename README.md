# Geometric Near-neighbor Access Tree (GNAT)

Python bindings for Geometric Near-neighbor Access Tree (GNAT) data structure implemented in OMPL. It can be used to search nearest neighbors in complex space such as SE3.

When dealing with complex customized space metric such as SE3 distance, GNAT has higher accuracy, takes less time to build, and has higher inference speed, comparing to common nearest neighbor package such as Balltree from scikit-learn. GNAT also supports online adding and removing data points.

Original C++ implementation:

[OMPL NearestNeighborsGNAT](https://ompl.kavrakilab.org/NearestNeighborsGNAT_8h_source.html)

Paper references:

[S. Brin, Near neighbor search in large metric spaces, in Proc. 21st Conf. on Very Large Databases (VLDB), pp. 574–584, 1995.](http://ilpubs.stanford.edu:8090/113/1/1995-44.pdf)

[B. Gipson, M. Moll, and L.E. Kavraki, Resolution independent density estimation for motion planning in high-dimensional spaces, in IEEE Intl. Conf. on Robotics and Automation, 2013.](https://moll.ai/publications/gipson2013resolution-independent-density-estimation.pdf)

## Running GNAT

Currently, GNAT supports data defined as a list of a 1-D numpy array. To build a GNAT, you first need to define a distance function by calling `set_distance_function()`. You may then add a list of data at once and build GNAT nearest neighbor structure by calling `add_list()`. Once GNAT is built, it provides nearest k neighbors search `nearest_k()` and nearest distance r search `nearest_r()`. Here is an example:

```
# Init GNAT
nn = gnat.NearestNeighborsGNAT()
nn.set_distance_function(distance_fn)
# Build GNAT
nn.add_list(data_points)

# Perform search
indices_k, nearest_k = nn.nearest_k(query_point, k)
indices_r, nearest_r = nn.nearest_r(query_point, r)
```

GNAT also supports online adding and removing a single data by calling `add()` and `remove()`. However, adding and removing data from a built GNAT may cause the tree to rebuild and can be inefficient (but will still be faster than rebuilding from scratch).

In order to locate the data in GNAT to be deleted, you can not simply pass the same data. Instead, data handle (pointer) will be returned by `add()` or `add_list()` to be used for data removal. For example:

```
data_handle = nn.add(data)
nn.remove(data_handle)
# nn.remove(data)  # this doesn't work
```

## Saving and loading GNAT

After GNAT is built with your data, you may save it locally to avoid building it again later. We provide two relevant functions `save()` and `load()`. GNAT is serialized as a pure string. So the suffix of your data file can be anything. You may simply run it as

```
# Build a GNAT
nn = gnat.NearestNeighborsGNAT()
nn.set_distance_function(se3_distance)
nn.add_list(data_points)

# Save GNAT
nn.save("gnat.dat")

# Load GNAT
nn2 = gnat.NearestNeighborsGNAT()
nn2.set_distance_function(se3_distance)
nn2.load("gnat.dat")
````

Please note that, when trying to load a GNAT from local serilized data, always call set_distance_function() first before loading. Calling set_distance_function() will rebuild the whole GNAT when data presented.

```
nn2 = gnat.NearestNeighborsGNAT()
# Run set_distance_function() first
nn2.set_distance_function(se3_distance)
nn2.load("gnat.dat")
````

## Complete example

For a complete example, please check the `test.py` and `test_online.py` script under **tests** folder.

To see the comparison with scikit-learn Balltree, please check the `comparison.py` script under **tests** folder.
