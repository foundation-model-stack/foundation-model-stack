# Model Testing

## Useful fixture mixins/test suites

fms comes with a set of useful test suites and fixture mixins which can be included for ease of testing:

Fixture Mixins:

- ConfigFixtureMixin - an abstract class which provides the user a ModelConfig fixture (requires implementation of config fixture)
- ModelFixtureMixin - an abstract class which provides the user a model fixture (requires implementation of uninitialized_model fixture)
- SignatureFixtureMixin - a class that provides the user with a signature fixture for a given model

Test Suites:

- ModelConfigTestSuite - This is a test suite which will test ModelConfigs and how they interact with the specific Model Architecture. Note: this will use ConfigFixtureMixin and ModelFixtureMixin
- ModelConsistencyTestSuite - This is a test suite which will test model consistency of output between versions. Note: this will use ModelFixtureMixin and SignatureFixtureMixin.

## Model Consistency Testing

In order to create your own model consistency testing suite, fms provides a set of scripts/mixins that can be used in automating model testing.
The following are the steps to follow to add a new model to the suite of generic model tests:

### 1. Create a Model Test Suite for your specific model

#### 1.1. Add a test file

Create a test file with the name test_<model_name>.py under tests/models

#### 1.2. Add a test class

Add the following code to the file, which will create a generic testing class for your model:

```python
class Test<model_name>(ModelConsistencyTestSuite):
    """
    Model Test Suite for <model_name>

    This suite will include tests for:
    - consistency of model output
    """
    
    # x is the main parameter for this model which is the input tensor
    _get_signature_params = ["x"]

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self):
        return <model_name>(...)
```
#### 1.3. Running the test suite

The first time you run the test suite, you will notice that the test will fail with the following message:

```txt
Signature failed to load, please re-run the tests with --capture_expectation

AND

Weights Key file failed to load, please re-run the tests with --capture_expectation
```

If you see this message, you will re-run the tests with the extra argument --capture_expectation, which will 
automatically include a signature resource for your model with the name <test_name>-Test<model_name>. The tests will
fail again now with the following messages:

```txt
Signature file has been saved, please re-run the tests without --capture_expectation

AND

Weights Key file has been saved, please re-run the tests without --capture_expectation
```

The next time you run the test, remove the --capture_expectation flag and the test should pass.