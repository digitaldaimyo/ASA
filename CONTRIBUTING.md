# Contributing

Thanks for your interest in contributing to ASA. This repository is a research codebase, so changes should stay small and reproducible.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Running tests

```bash
pytest
```

## Style expectations

- Keep changes focused and minimal.
- Match existing code style and formatting.
- Avoid introducing new linting or formatting dependencies.

## Pull requests

- Explain the motivation and scope of the change.
- Include tests when behavior changes or new features are added.
- Keep PRs small and easy to review.

## Issues

When filing issues, include:

- A short description of the problem.
- Steps to reproduce.
- Expected vs. actual behavior.
- Environment details (OS, Python version, relevant dependencies).
