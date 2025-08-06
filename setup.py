import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

description = (
    "Reinforcement Learning toolbox for BabyBench2025"
)

setuptools.setup(
    name="babybench25",
    version="0.1.0.dev0",
    author="Marco Gabriele Fedozzi, Leticia Berto",
    author_email="marco.fedozzi@iit.it,leticia.berto@postgrad.manchester.ac.uk",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/marcogfedozzi/BabyBench2025_Starter_Kit",
    packages=setuptools.find_packages(where='babybench', exclude=["tests", "tests.*"]),
    package_dir={"": "babybench"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9', 
    install_requires=[]
)