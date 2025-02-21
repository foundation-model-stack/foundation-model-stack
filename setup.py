from setuptools import find_packages, setup


setup(
    name="ibm-fms",
    version="0.0.8",
    author="Brian Vaughan, Joshua Rosenkranz, Antoni Viros i Martin, Davis Wertheimer, Supriyo Chakraborty, Raghu Kiran Ganti",
    author_email="bvaughan@ibm.com, jmrosenk@us.ibm.com, aviros@ibm.com, Davis.Wertheimer@ibm.com, supriyo@us.ibm.com, rganti@us.ibm.com",
    description="IBM Foundation Model Stack",
    url="https://github.com/foundation-model-stack/foundation-model-stack",
    packages=find_packages(),
    install_requires=["torch >= 2.5.1"],
    extras_require={"hf": ["transformers >= 4.48.3"]},
    license="Apache License 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    package_data={"ibm-fms": ["py.typed"]},
)
