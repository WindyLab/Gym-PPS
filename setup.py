import os.path
import sys

from setuptools import find_packages, setup

# Don't import gym module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gym"))
from version import VERSION 

# Environment-specific dependencies.
extras = {
    "atari": ["atari-py==0.2.6", "opencv-python>=3."],
    "box2d": ["box2d-py~=2.3.5", "pyglet>=1.4.0"],
    "classic_control": ["pyglet==1.5.27"],
    "mujoco": ["mujoco_py>=1.50, <2.0"],
    "robotics": ["mujoco_py>=1.50, <2.0"],
    "toy_text": ["scipy>=1.4.1"],
    "other": ["lz4>=3.1.0", "opencv-python>=3."],
}

# Meta dependency groups.
extras["nomujoco"] = list(
    set(
        [
            item
            for name, group in extras.items()
            if name != "mujoco" and name != "robotics"
            for item in group
        ]
    )
)
extras["all"] = list(set([item for group in extras.values() for item in group])) 


setup(
    name="gym",
    version=VERSION,
    description="Gym: A universal API for reinforcement learning environments.",
    url="https://github.com/openai/gym",
    author="OpenAI",
    author_email="jkterry@umd.edu",
    license="",
    packages=[package for package in find_packages() if package.startswith("gym")],
    zip_safe=False,
    install_requires=[
        "numpy==1.18.0",
        "cloudpickle>=1.2.0",
        "setuptools==57.5.0",
        "pyglet==1.5.27"
    ],
    extras_require=extras,
    package_data={
        "gym": [
            "envs/mujoco/assets/*.xml", 
            "envs/classic_control/assets/*.png", 
            "envs/robotics/assets/LICENSE.md",
            "envs/robotics/assets/fetch/*.xml",
            "envs/robotics/assets/hand/*.xml",
            "envs/robotics/assets/stls/fetch/*.stl",
            "envs/robotics/assets/stls/hand/*.stl",
            "envs/robotics/assets/textures/*.png",
            "utils/*.so",
        ]
    },
    tests_require=["pytest", "mock"],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)

