#!/usr/bin/env bash

python setup.py bdist_wheel
twine upload -u ${ARTIFACTORY_USERNAME} -p ${ARTIFACTORY_API_KEY} dist/*