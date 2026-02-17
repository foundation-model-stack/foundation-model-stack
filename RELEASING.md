# Release Process

Currently, we only have a single release process for pushing releases off `main`.

## Main Release Process
1. **Create a Tag**
   - Navigate to the repository's main branch.
   - Create a new tag adhering to the convention `vX.Y.Z`, where `X.Y.Z` represents the new version number.
   - Push the newly created tag to the remote repository.

2. **Create a Release**
   - Go to the 'Releases' section of your GitHub repository.
   - Click on 'Draft a new release'.
   - Select the tag you just pushed from the 'Release a new tag' dropdown.
   - Utilize the 'Generate release notes' feature to automatically populate notes based on changes since the last release.

3. **Prepare for Publication**
   - Review and finalize the drafted release notes.
   - Ensure all necessary files and metadata are included.

4. **Publish the Release**
   - Once satisfied, publish the release by clicking the 'Publish release' button.

5. **Automated Publishing to PyPi**
   - Upon publishing, the `build-and-publish.yaml` workflow will be triggered.
   - This workflow will execute, constructing a new wheel file, and subsequently push it to [PyPi](https://pypi.org/project/ibm-fms/).

## Optional Publishing on TestPyPi

This action will publish to TestPyPi for quality assurance before final publishing to PyPi:

1. Access the [build-and-publish workflow](https://github.com/foundation-model-stack/foundation-model-stack/actions/workflows/build-and-publish.yaml) within GitHub Actions.
2. From the 'Run workflow' dropdown, select the desired branch (`main`) and specify a pre-release tag, such as `v1.3.0.rc1`.
3. Manually activate the workflow through the 'Run workflow' button.
4. For TestPyPi package installation:

    ```shell
    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/project ibm-fms==v1.3.0rc1
    ```
