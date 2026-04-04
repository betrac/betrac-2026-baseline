# Contributing to Beyond Transcription Challenge Baseline

Thank you for your interest in contributing to the Beyond Transcription Challenge baseline! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Linting and Formatting](#linting-and-formatting)
- [Submitting Changes](#submitting-changes)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Code Review Process](#code-review-process)
- [Project Structure](#project-structure)

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11** (strict requirement - see `.python-version`)
- **uv** - Fast Python package manager ([installation guide](https://github.com/astral-sh/uv))
- **Git** - Version control
- **Make** - Build automation (usually pre-installed on Unix systems)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/betrac-2026-baseline.git
   cd betrac-2026-baseline
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/ORIGINAL-OWNER/betrac-2026-baseline.git
   ```

### Understand the Codebase

Before making changes, familiarize yourself with:

- [README.md](README.md) - Project overview and quick start
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and design decisions
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions

## Development Setup

### 1. Create Virtual Environment

```bash
make setup
```

This creates `steps/omni/.venv` and installs all production dependencies.

### 2. Install Dev Tools

```bash
cd steps/omni && uv pip install pytest ruff mypy
```

### 3. Verify Installation

```bash
make test     # Smoke test with example audio
make lint     # Run linter and type checker
```

### 4. (Optional) Install Pre-commit Hooks

For automatic code quality checks on every commit:

```bash
pip install pre-commit && pre-commit install
```

This runs before each `git commit`:
- Ruff linting and formatting
- YAML/TOML syntax validation
- File quality checks (trailing whitespace, large files, etc.)

**How it works:**

Once installed, every time you run `git commit`, pre-commit automatically:
1. Runs all configured checks on your staged files
2. Auto-fixes issues when possible (formatting, import sorting)
3. **Blocks the commit if checks fail** ❌
4. **Allows the commit if checks pass** ✅

**Example workflow:**

```bash
# Make changes to code
vim steps/omni/run_omni.py

# Stage changes
git add steps/omni/run_omni.py

# Try to commit
git commit -m "Add new feature"

# Pre-commit runs automatically:
Ruff....................................................................Passed
Ruff Format.............................................................Passed
mypy....................................................................Passed
Trim Trailing Whitespace................................................Passed
Fix End of Files........................................................Passed
Check Yaml..............................................................Passed

# ✅ Commit succeeds!
```

**Manual usage:**

```bash
# Run on all files without committing
pre-commit run --all-files

# Run on specific files
pre-commit run --files steps/omni/run_omni.py

# Update to latest hook versions
pre-commit autoupdate
```

**Skip pre-commit for urgent fixes:**

```bash
# Use sparingly - only for emergencies
git commit --no-verify -m "Emergency fix"
```

**Don't want pre-commit hooks?**

That's fine! You can skip this step and run checks manually before committing:

```bash
make check  # Runs linting and tests manually
```

## Code Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with the following specifics:

- **Maximum line length**: 88 characters (enforced by ruff formatter)
- **Indentation**: 4 spaces (no tabs)
- **Quote style**: Double quotes for strings
- **Import order**: Standard library → Third-party → Local (managed by ruff)

### Type Hints

- **Required for new code**: All new functions must include type hints
- **Function signatures**: Include parameter types and return types
- **Use typing module**: For complex types (List, Dict, Optional, etc.)

Example:
```python
from typing import List, Optional

def process_files(file_paths: List[str], output_dir: str) -> Optional[int]:
    """Process a list of files and return count of successful operations.

    Args:
        file_paths: List of paths to files to process
        output_dir: Directory where outputs will be written

    Returns:
        Number of successfully processed files, or None if all failed
    """
    # Implementation
    pass
```

### Documentation

- **Docstrings required**: All functions, classes, and modules
- **Format**: Google-style docstrings
- **Include**:
  - Brief description
  - Args section
  - Returns section
  - Raises section (if applicable)

### Code Organization

- **One concept per function**: Keep functions focused and single-purpose
- **Meaningful names**: Use descriptive variable and function names
- **Comments for complex logic**: Explain "why", not "what"
- **Avoid magic numbers**: Use named constants

## Testing Guidelines

### Writing Tests

- **Test file naming**: `test_*.py` (e.g., `test_steps/omni/run_omni.py`)
- **Test function naming**: `test_*` (e.g., `test_memory_logging`)
- **Test class naming**: `Test*` (e.g., `TestMemoryLogging`)

### Test Structure

Use the Arrange-Act-Assert pattern:

```python
def test_function_name():
    # Arrange: Set up test data and conditions
    input_data = create_test_input()

    # Act: Execute the function being tested
    result = function_under_test(input_data)

    # Assert: Verify the result
    assert result == expected_output
```

### Test Types

1. **Unit Tests** (fast, isolated)
   - Test individual functions
   - Use mocks for external dependencies
   - Should complete in < 1 second each
   - Run with: `pytest tests/ -v -m "not integration"`

2. **Smoke Tests** (requires model + HuggingFace access)
   - `make test` — run on example manifest (short audio)
   - `make test-5` — run on 5 HuggingFace validation samples (uses Qwen2.5-Omni-3B)
   - `make test-hf` — run on HuggingFace samples (configurable MODEL, LIMIT)

3. **Integration Tests** (slower, requires full setup)
   - Test complete workflows
   - May require model files and GPU
   - Mark with `@pytest.mark.integration`
   - Run with: `pytest tests/ -v -m integration`

### Coverage Requirements

- **Aim for 80%+ coverage** for new code
- Run coverage report: `pytest tests/ --cov=run_omni`
- View HTML report: Open `htmlcov/index.html`

### Testing Best Practices

- **Keep tests independent**: Each test should run in isolation
- **Use fixtures**: Share setup code with pytest fixtures (see `tests/conftest.py`)
- **Test edge cases**: Include boundary conditions and error cases
- **Descriptive assertions**: Use clear assertion messages

## Linting and Formatting

### Running Linters

```bash
# Check for issues (read-only)
make lint

# Auto-fix issues and format code
make lint-fix

# Format code only
make format
```

### Pre-Commit Checklist

Before committing, run:

```bash
make lint
pytest tests/ -v -m "not integration"
```

## Submitting Changes

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions or modifications

### 2. Make Your Changes

- Write clean, well-documented code
- Add tests for new functionality
- Update documentation as needed
- Follow the code standards above

### 3. Test Your Changes

```bash
make lint
pytest tests/ -v -m "not integration"
```

### 4. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add feature: brief description

Longer explanation of what changed and why. Reference any
related issues (e.g., Fixes #123, Closes #456).

- Bullet points for specific changes
- Keep lines under 72 characters
- Use present tense (Add, Fix, Update)"
```

### 5. Keep Your Branch Updated

```bash
# Fetch latest changes from upstream
git fetch upstream

# Rebase your branch on upstream main
git rebase upstream/main
```

### 6. Push Your Changes

```bash
git push origin feature/your-feature-name
```

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows the style guidelines
- [ ] All tests pass (`make test`)
- [ ] Linting passes (`make lint`)
- [ ] Coverage maintained or improved
- [ ] Documentation updated
- [ ] CHANGELOG updated (if applicable)

### Creating a Pull Request

1. **Navigate to your fork** on GitHub
2. **Click "New Pull Request"**
3. **Provide a clear title** (e.g., "Add audio preprocessing module")
4. **Fill out the description template**:
   ```markdown
   ## Description
   Brief summary of changes

   ## Motivation and Context
   Why is this change needed? What problem does it solve?

   ## Type of Change
   - [ ] Bug fix (non-breaking change fixing an issue)
   - [ ] New feature (non-breaking change adding functionality)
   - [ ] Breaking change (fix or feature causing existing functionality to change)
   - [ ] Documentation update

   ## Testing
   - How was this tested?
   - What tests were added?

   ## Screenshots (if applicable)

   ## Related Issues
   Fixes #123, Closes #456
   ```

5. **Link related issues** using keywords (Fixes, Closes, Resolves)
6. **Request reviewers** if you know appropriate maintainers

### Pull Request Review

- Expect constructive feedback
- Address comments promptly
- Update your PR based on feedback
- Use `git commit --amend` or new commits as appropriate
- Keep discussion professional and focused

## Code Review Process

### For Contributors

- **Be responsive**: Reply to review comments within a few days
- **Ask questions**: If feedback is unclear, ask for clarification
- **Test suggestions**: Verify that suggested changes work
- **Be patient**: Reviews may take time depending on maintainer availability

### Review Criteria

Reviewers will check:

1. **Functionality**: Does the code work as intended?
2. **Tests**: Are there adequate tests? Do they pass?
3. **Code quality**: Is the code clean, readable, maintainable?
4. **Documentation**: Are changes documented appropriately?
5. **Style**: Does code follow project conventions?
6. **Performance**: Are there any obvious performance issues?
7. **Security**: Are there potential security concerns?

### Merging

Once approved:
- Maintainers will merge your PR
- Your changes will be included in the next release
- You'll be credited as a contributor!

## Project Structure

```
betrac-2026-baseline/
├── steps/omni/
│   ├── run_omni.py            # Main inference script
│   └── pyproject.toml         # Pinned dependencies
├── conf/prompt/
│   └── default.yaml           # SOAP note prompt template
├── scripts/                    # SLURM templates and utilities
├── tests/                      # Test suite
├── data/manifests/             # CSV manifests for custom data
├── examples/                   # Example outputs
├── pyproject.toml             # Project metadata and tool config
├── Makefile                   # Build and run automation
└── *.md                       # Documentation
```

### Key Files

- **steps/omni/run_omni.py**: Model loading, inference, output
- **conf/prompt/default.yaml**: SOAP note generation prompt
- **pyproject.toml**: Tool config (ruff, mypy, pytest)
- **steps/omni/pyproject.toml**: Pinned runtime dependencies
- **Makefile**: Build, run, and test automation

## Questions or Need Help?

- **Documentation**: Check [README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md), [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Issues**: Search existing [GitHub Issues](../../issues)
- **New Issue**: Open a new issue with details
- **Discussions**: Use [GitHub Discussions](../../discussions) for questions

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions
- Keep discussions professional

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

## Recognition

Contributors will be:
- Listed in the project's contributors
- Credited in release notes
- Appreciated by the community!

---

**Thank you for contributing to Beyond Transcription Challenge Baseline!** 🎉

Your contributions help advance research in clinical AI and medical documentation automation.
