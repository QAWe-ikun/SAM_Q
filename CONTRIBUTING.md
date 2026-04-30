# Contributing to SAM-Q

Thank you for your interest in contributing to SAM-Q! This guide provides guidelines and best practices for contributing.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Documentation](#documentation)
6. [Pull Request Process](#pull-request-process)

---

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Prioritize what's best for the community

---

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/SAM_Q.git
cd SAM_Q
git remote add upstream https://github.com/ORIGINAL_OWNER/SAM_Q.git
```

### 2. Set Up Environment

```bash
# Create virtual environment
conda create -n samq-dev python=3.10
conda activate samq-dev

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

### 3. Verify Installation

```bash
# Run tests
python -m pytest tests/ -v

# Check code style
black --check src/
flake8 src/
```

---

## Development Workflow

### Branch Naming

Use descriptive branch names:

```
feature/add-transformer-encoder
fix/memory-leak-inference
docs/update-readme
refactor/adapter-modules
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples**:
```
feat(adapter): add presence token adapter
fix(inference): resolve memory leak in predictor
docs(readme): update installation instructions
refactor(models): modularize encoder components
```

### Git Workflow

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/my-feature

# Make changes and commit
git add .
git commit -m "feat(scope): description"

# Push and create PR
git push origin feature/my-feature
```

---

## Coding Standards

### Python Style Guide

We follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) with these highlights:

#### 1. Formatting

Use `black` for automatic formatting:

```bash
black src/ tests/
```

#### 2. Type Hints

All functions must have type hints:

```python
def compute_iou(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute IoU score."""
    ...
```

#### 3. Docstrings

Use Google-style docstrings:

```python
class MyModel(nn.Module):
    """
    Short description.
    
    Longer description if needed.
    
    Args:
        param1: Description
        param2: Description
    
    Returns:
        Description of return value
    """
```

#### 4. Naming Conventions

- **Modules**: `snake_case` (`cross_modal_adapter.py`)
- **Classes**: `PascalCase` (`CrossModalAdapter`)
- **Functions/Variables**: `snake_case` (`compute_iou`)
- **Constants**: `UPPER_SNAKE_CASE` (`MAX_BATCH_SIZE`)

---

## Documentation

### Updating Documentation

When adding features:

1. Update `README.md` if user-facing
2. Update `ARCHITECTURE.md` if architecture changes
3. Add docstrings to all public APIs
4. Add examples in docstrings

---

## Pull Request Process

### Before Submitting

1. **Update tests**: Ensure all tests pass
2. **Update docs**: Documentation is current
3. **Run linters**: Code follows style guide
4. **Rebase on main**: Latest changes incorporated

### PR Checklist

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Added unit tests
- [ ] All tests pass
- [ ] Manual testing completed

## Documentation
- [ ] README updated
- [ ] Docstrings added/updated
- [ ] ARCHITECTURE.md updated (if applicable)
```

### Review Process

1. CI checks run automatically
2. At least one maintainer review required
3. Address review comments
4. Squash commits if requested
5. Maintainer merges PR

---

## Common Contributions

### Adding a New Dataset

1. Create file: `src/data/my_dataset.py`
2. Inherit from `torch.utils.data.Dataset`
3. Implement `__len__`, `__getitem__`
4. Add to `src/data/__init__.py`
5. Add example to docs

### Auto-Generating Training Data

If you have 3D scene data (e.g., SSR3D-FRONT):

1. Use `src/pretreatment/data_generator.py`
2. Configure scene/model paths and output parameters
3. Run with `--augmentation` for data augmentation
4. Outputs: `splits.json`, `plane_images/`, `object_images/`, `masks/`

See [docs/DATA_GENERATION.md](docs/DATA_GENERATION.md) for details.

### Fixing a Bug

1. Write test that reproduces bug
2. Fix the issue
3. Verify test passes
4. Submit PR with description

---

## Getting Help

- **Questions**: Open an issue with label `question`
- **Bugs**: Open an issue with label `bug`
- **Features**: Open an issue with label `enhancement`
- **Chat**: Join our Discord (link in README)

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to SAM-Q! 🎉
