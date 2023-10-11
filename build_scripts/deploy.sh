#!/usr/bin/env bash

rm -rf dist || true
python setup.py bdist_wheel
twine upload dist/*