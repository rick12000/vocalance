---
name: pytest-unit-testing
description: >
  Use when writing or reviewing Python unit tests, pytest test suites, pytest fixtures, or unittest.mock usage.
  Handles test target selection, fixture placement, parametrization, mocking boundaries, shaped outputs, mathematical checks, and test failure triage.
  Do NOT use for non-Python test frameworks or broad QA strategy unless pytest unit tests are part of the task.
---

# Pytest Unit Testing

## Workflow

1. Use `pytest` for all tests. Use `unittest.mock` for mocking when needed.
2. Gather context from the codebase for the classes, methods, or functions the user requested tests for.
3. Gather context on surrounding classes, methods, or functions to understand intent, dependencies, behaviour, and expected usage.
4. Create a short list of classes, methods, or functions worth unit testing. Include targets where tests add value through documentation, error catching, or edge case coverage that exceeds their maintenance burden and code clutter. Prefer central functionality. Skip targets that are non-central, extremely straightforward, mostly initialization, or heavily dependent on external APIs, network calls, or I/O.
5. Write one happy-path test for each selected target.
6. Identify which selected targets benefit from edge case tests. Add edge tests only when their value exceeds maintenance burden and clutter.
7. Write tests according to the rules below.
8. Run the tests.
9. Review failures. Decide whether each failure is likely caused by broken production code or by test infrastructure problems such as bad setup, incorrect assumptions, weak fixtures, broken `conftest.py`, or bad import paths.
10. Fix test infrastructure problems and rerun until no failures are purely caused by the tests.
11. Report remaining failures as likely production-code issues revealed by the unit tests.

## Rules

- Use `pytest.mark.parametrize` for categorical input values. For `Literal[...]`, parametrize over every possible literal value. For ordinal or discrete inputs such as `n_observations` or `n_recommendations`, use sensible ranges: `0` if allowed, otherwise the minimum; a normal value such as `10`; and a large cheap value such as `1000`.
- For multiple parametrized variables, use one `@pytest.mark.parametrize` decorator per variable. Stacked parametrization produces the exhaustive Cartesian product of all provided values.

  ```python
  @pytest.mark.parametrize("metric", ["cosine", "euclidean"])
  @pytest.mark.parametrize("n_recommendations", [1, 10, 1000])
  def test_recommendations(metric, n_recommendations):
      ...
  ```

  This example produces six test cases:

  ```
  ("cosine", 1)
  ("cosine", 10)
  ("cosine", 1000)
  ("euclidean", 1)
  ("euclidean", 10)
  ("euclidean", 1000)
  ```

- Mock external APIs and I/O only. Do not use mocking to abstract away components whose behaviour needs to be tested.
- Use fixtures for reusable toy data, mocked objects, shared expected values, or any object referenced in more than one test. Define all fixtures in `tests/conftest.py`, never directly in a test module. Define tiny one-off toy data inside the test.
- Never define nested functions unless scope requires it, such as nested generator builders.
- Avoid top-level helper functions in test modules. Prefer simple tests that call existing code. Add helpers only when a test would otherwise become complex or repetitive.
- Do not test class initialization. Do not assert only that an attribute exists or equals the value just passed into initialization.
- For shaped outputs, explicitly assert shape-related properties such as length, dimensions, row counts, column counts, tensor shapes, dataframe shapes, collection sizes, or whether the output shape should match or differ from the input shape.
- Test the intent behind a function or method, not its implementation details, attributes, or internal structure. Understand what the code is trying to achieve and validate that behaviour.
- Do not add assertion messages:

  ```python
  assert len(result) == expected_length
  ```

  not:

  ```python
  assert len(result) == expected_length, "Unexpected result length"
  ```

- Keep comments rare and only use them to explain non-obvious assertions or test scenarios.
- Each unit test should be a standalone function. Do not use test classes or `self`.
- For mathematical functions, understand the derivations and test assumptions, invariants, constraints, theoretical properties, and expected outputs.
- Keep the test suite lean. Prioritize tests that provide meaningful documentation, regression protection, or edge-case coverage relative to their maintenance cost.
- Avoid repetitive tests. When multiple tests share the same setup and only differ in assertions, combine them into a single behaviour-focused test where appropriate.
- Do not use `try`/`except` in tests. Tests should fail with their original traceback.
