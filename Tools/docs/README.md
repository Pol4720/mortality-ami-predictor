# Documentation Site for Mortality AMI Predictor

This directory contains the complete documentation for the Mortality AMI Predictor project.

## 🚀 Quick Start

### Install Documentation Dependencies

```bash
pip install -r docs-requirements.txt
```

### Build Documentation

```bash
# Build static site
python build_docs.py

# Build and serve locally
python build_docs.py --serve

# Serve on custom port
python build_docs.py --serve --port 8080
```

### Using MkDocs Directly

```bash
# Build
mkdocs build

# Serve locally
mkdocs serve

# Deploy to GitHub Pages
mkdocs gh-deploy
```

## 📁 Documentation Structure

```
docs/
├── index.md                    # Home page
├── getting-started/           # Installation and quick start
│   ├── installation.md
│   ├── quickstart.md
│   └── configuration.md
├── user-guide/                # Feature guides
│   ├── dashboard.md
│   ├── data-cleaning.md
│   ├── eda.md
│   ├── training.md
│   ├── predictions.md
│   ├── evaluation.md
│   ├── explainability.md
│   ├── clinical-scores.md
│   └── custom-models.md
├── api/                       # API reference
│   ├── index.md
│   ├── config.md
│   ├── data-load/
│   ├── cleaning/
│   ├── eda/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   ├── training/
│   ├── prediction/
│   ├── evaluation/
│   ├── explainability/
│   ├── scoring/
│   └── reporting/
├── architecture/              # Design documentation
│   ├── patterns.md
│   ├── structure.md
│   ├── custom-models.md
│   └── data-flow.md
├── developer/                 # Developer guides
│   ├── contributing.md
│   ├── testing.md
│   ├── style.md
│   └── migration.md
├── assets/                    # Images and resources
│   ├── logo.png
│   └── favicon.ico
├── stylesheets/              # Custom CSS
│   └── extra.css
└── javascripts/              # Custom JavaScript
    └── extra.js
```

## 🎨 Features

### Built-in Features

- ✅ **Automatic API documentation** from docstrings using mkdocstrings
- ✅ **Search functionality** with instant results
- ✅ **Dark/Light mode** toggle
- ✅ **Mobile responsive** design
- ✅ **Code syntax highlighting** with copy button
- ✅ **Mermaid diagrams** support
- ✅ **Math equations** with KaTeX
- ✅ **Navigation breadcrumbs**
- ✅ **Table of contents** with auto-scrolling
- ✅ **Git revision dates** on pages
- ✅ **Social links** to GitHub
- ✅ **Keyboard shortcuts** (Ctrl+K for search)

### Custom Features

- ✅ **Custom logo** with heartbeat animation
- ✅ **Branded colors** matching the project theme
- ✅ **Progress bar** for reading
- ✅ **Enhanced code blocks** with language labels
- ✅ **Smooth scrolling** and animations
- ✅ **External link icons** automatically added
- ✅ **Print optimization**

## 📝 Writing Documentation

### Adding a New Page

1. Create a new `.md` file in the appropriate directory
2. Add it to `mkdocs.yml` navigation
3. Write content using Markdown

Example:

```markdown
# My New Page

This is the content of my page.

## Section 1

Some content here.

## Section 2

More content.
```

### Documenting Python Code

Use Google-style docstrings:

```python
def my_function(param1: str, param2: int) -> bool:
    """Brief description of the function.
    
    Longer description with more details about what the function does.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When param2 is negative
    
    Example:
        >>> result = my_function("test", 42)
        >>> print(result)
        True
    """
    return True
```

Then reference it in documentation:

```markdown
# My Module

::: src.mymodule.my_function
    options:
      show_source: true
```

### Using Admonitions

```markdown
!!! note
    This is a note admonition.

!!! tip "Custom Title"
    This is a tip with custom title.

!!! warning
    This is a warning.

!!! danger
    This is a danger alert.

!!! example
    This is an example.
```

### Code Blocks

````markdown
```python
def hello():
    print("Hello, World!")
```

```bash
# Install dependencies
pip install -r requirements.txt
```
````

### Tabs

```markdown
=== "Python"

    ```python
    print("Hello")
    ```

=== "JavaScript"

    ```javascript
    console.log("Hello");
    ```
```

### Mermaid Diagrams

````markdown
```mermaid
graph LR
    A[Start] --> B[Process]
    B --> C[End]
```
````

### Math Equations

```markdown
Inline math: $E = mc^2$

Block math:

$$
\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
$$
```

## 🎨 Customization

### Colors

Edit `docs/stylesheets/extra.css` to change colors:

```css
:root {
  --md-primary-fg-color: #b71c1c;  /* Primary color */
  --md-accent-fg-color: #ff5722;    /* Accent color */
}
```

### JavaScript

Add custom JavaScript in `docs/javascripts/extra.js`.

### Logo

Replace `docs/assets/logo.png` with your logo (recommended: 500x500px PNG).

## 🚀 Deployment

### GitHub Pages

```bash
# Deploy to GitHub Pages
python build_docs.py --deploy

# Or using mkdocs directly
mkdocs gh-deploy
```

### Custom Server

```bash
# Build static site
mkdocs build

# Copy 'site' directory to your web server
scp -r site/* user@server:/var/www/docs/
```

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /docs

COPY docs-requirements.txt .
RUN pip install -r docs-requirements.txt

COPY mkdocs.yml .
COPY docs/ docs/
COPY src/ src/

RUN mkdocs build

# Serve with nginx or python http server
CMD ["python", "-m", "http.server", "8000", "--directory", "site"]
```

## 📊 Documentation Metrics

- **Total pages**: 40+
- **API reference pages**: 30+
- **Code examples**: 100+
- **Diagrams**: 10+
- **Languages**: English (Spanish support ready)

## 🤝 Contributing

When contributing to documentation:

1. Follow the existing structure and style
2. Add examples for all API functions
3. Use proper Markdown formatting
4. Test locally before committing
5. Update navigation in `mkdocs.yml` if adding new pages

## 📚 Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [mkdocstrings](https://mkdocstrings.github.io/)
- [Python Markdown Extensions](https://facelessuser.github.io/pymdown-extensions/)

## 🆘 Troubleshooting

### Build Errors

```bash
# Clear cache and rebuild
rm -rf site/
mkdocs build --clean
```

### Import Errors

Make sure you're in the Tools directory and src is importable:

```bash
cd Tools
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
mkdocs build
```

### Port Already in Use

```bash
# Use different port
mkdocs serve --dev-addr localhost:8001
```

## 📄 License

This documentation is part of the Mortality AMI Predictor project and follows the same license (MIT).
