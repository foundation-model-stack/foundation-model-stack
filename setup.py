from setuptools import find_packages, setup


setup(
    name="fms",
    version="0.0.1",
    author="Brian Vaughan, Joshua Rosenkranz, Antoni Viros i Martin, Davis Wertheimer, Supriyo Chakraborty, Raghu Kiran Ganti",
    author_email="bvaughan@ibm.com, jmrosenk@us.ibm.com, aviros@ibm.com, Davis.Wertheimer@ibm.com, supriyo@us.ibm.com, rganti@us.ibm.com",
    description="IBM Foundation Model Stack",
    packages=find_packages(),
    install_requires=["torch >= 2.1"],
    extras_require={"hf": ["transformers >= 4.34.0"]},
    license="Apache License 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)
