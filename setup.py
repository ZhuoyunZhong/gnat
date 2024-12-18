# setup.py
from skbuild import setup
from setuptools import find_packages
import sys

# Determine the CMake generator based on the operating system
if sys.platform == "win32":
    cmake_generator = "NMake Makefiles"
elif sys.platform == "darwin":  # macOS
    cmake_generator = "Unix Makefiles"
else:  # Linux and other platforms
    cmake_generator = "Unix Makefiles"

setup(
    name="gnat",
    version="0.2.0",
    description="Python bindings for the GNAT data structure",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Zhuoyun Zhong",
    author_email="zzy905954450@gmail.com",
    url="https://github.com/ZhuoyunZhong/gnat",
    packages=find_packages(),
    package_dir={"gnat": "gnat"},
    cmake_install_dir="gnat",
    include_package_data=True,
    python_requires=">=3.8",
    cmake_args=["-G", cmake_generator],
    install_requires=["pybind11", "scikit-build"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
)
