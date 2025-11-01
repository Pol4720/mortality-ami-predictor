# 🫀 Mortality AMI Predictor - Documentation System

Beautiful, interactive documentation powered by **MkDocs Material** with automatic API reference generation.

## ✨ Features

- 📚 **Auto-generated API docs** from Python docstrings
- 🎨 **Beautiful Material Design** theme with custom branding
- 🔍 **Advanced search** with instant results
- 🌓 **Dark/Light mode** toggle
- 📱 **Mobile responsive** design
- 🚀 **Fast and modern** static site
- 🎯 **Interactive examples** with syntax highlighting
- 📊 **Mermaid diagrams** support
- ⚡ **Live reload** during development

## 🚀 Quick Start

### 1. Install Documentation Dependencies

```powershell
# In the Tools directory
cd Tools
pip install -r docs-requirements.txt
```

### 2. Serve Documentation Locally

```powershell
# Using the build script (recommended)
python build_docs.py --serve

# Or using mkdocs directly
mkdocs serve
```

The documentation will be available at: `http://localhost:8000`

### 3. Generate API Documentation

Auto-generate API reference pages from source code:

```powershell
python generate_api_docs.py
```

This will create documentation pages for all modules in `src/`.

## 📁 Documentation Structure

```
docs/
├── index.md                      # Home page with overview
├── getting-started/              # Installation and setup
│   ├── installation.md          # Step-by-step installation
│   ├── quickstart.md            # 5-minute quick start
│   └── configuration.md         # Configuration guide
├── user-guide/                   # Feature documentation
│   ├── dashboard.md             # Dashboard overview
│   ├── data-cleaning.md         # Data cleaning guide
│   ├── eda.md                   # EDA guide
│   ├── training.md              # Model training
│   ├── predictions.md           # Making predictions
│   ├── evaluation.md            # Model evaluation
│   ├── explainability.md        # Model interpretation
│   ├── clinical-scores.md       # Clinical scoring
│   └── custom-models.md         # Custom model creation
├── api/                          # Auto-generated API reference
│   ├── config.md                # Configuration module
│   ├── cleaning/                # Cleaning module docs
│   ├── data-load/               # Data loading docs
│   ├── eda/                     # EDA module docs
│   ├── training/                # Training module docs
│   ├── evaluation/              # Evaluation module docs
│   └── ...                      # Other modules
├── architecture/                 # Design documentation
│   ├── patterns.md              # Design patterns
│   ├── structure.md             # Module structure
│   ├── custom-models.md         # Custom models architecture
│   └── data-flow.md             # Data flow diagrams
├── developer/                    # Developer guides
│   ├── contributing.md          # How to contribute
│   ├── testing.md               # Testing guide
│   ├── style.md                 # Code style guide
│   └── migration.md             # Migration from v1 to v2
├── assets/                       # Images and resources
│   ├── logo.png                 # Project logo
│   └── favicon.ico              # Site favicon
├── stylesheets/                  # Custom CSS
│   └── extra.css                # Theme customization
└── javascripts/                  # Custom JavaScript
    └── extra.js                 # Enhanced functionality
```

## 🎨 Customization

### Logo and Branding

1. Place your logo in `docs/assets/logo.png` (recommended: 500x500px PNG)
2. Update colors in `docs/stylesheets/extra.css`:

```css
:root {
  --md-primary-fg-color: #b71c1c;  /* Primary red */
  --md-accent-fg-color: #ff5722;    /* Accent orange */
}
```

### Theme Configuration

Edit `mkdocs.yml` to customize:

```yaml
theme:
  name: material
  palette:
    - scheme: default
      primary: red
      accent: deep orange
```

## 📝 Writing Documentation

### Adding a New Page

1. Create a `.md` file in the appropriate directory
2. Add to navigation in `mkdocs.yml`:

```yaml
nav:
  - User Guide:
    - My New Page: user-guide/my-new-page.md
```

### Documenting Python Code

Use Google-style docstrings:

```python
def my_function(param1: str, param2: int = 10) -> bool:
    """Brief description of the function.
    
    Longer description with more details.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)
    
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

Then reference it:

```markdown
::: src.mymodule.my_function
    options:
      show_source: true
```

### Using Markdown Features

**Admonitions:**
```markdown
!!! tip "Pro Tip"
    This is a helpful tip!

!!! warning
    This is a warning.

!!! example
    This is an example.
```

**Code Tabs:**
```markdown
=== "Python"
    ```python
    print("Hello")
    ```

=== "Bash"
    ```bash
    echo "Hello"
    ```
```

**Mermaid Diagrams:**
````markdown
```mermaid
graph LR
    A[Start] --> B[Process]
    B --> C[End]
```
````

## 🚀 Building and Deployment

### Build Static Site

```powershell
# Using build script
python build_docs.py

# Or using mkdocs
mkdocs build
```

Output will be in the `site/` directory.

### Deploy to GitHub Pages

```powershell
# Deploy automatically
python build_docs.py --deploy

# Or manually
mkdocs gh-deploy --force
```

The documentation will be live at: `https://pol4720.github.io/mortality-ami-predictor/`

### Deploy to Custom Server

```powershell
# Build site
mkdocs build

# Copy to server
scp -r site/* user@server:/var/www/docs/
```

## 🔧 Commands

| Command | Description |
|---------|-------------|
| `python build_docs.py --serve` | Build and serve locally |
| `python build_docs.py --deploy` | Deploy to GitHub Pages |
| `python generate_api_docs.py` | Generate API reference pages |
| `mkdocs build` | Build static site |
| `mkdocs serve` | Serve locally with live reload |
| `mkdocs gh-deploy` | Deploy to GitHub Pages |

## 🎯 Features Included

### Built-in Features

✅ **Search** - Instant search across all pages  
✅ **Navigation** - Tabs, sections, and breadcrumbs  
✅ **TOC** - Table of contents with auto-scroll  
✅ **Code** - Syntax highlighting with copy button  
✅ **Responsive** - Works on all devices  
✅ **Git Integration** - Shows last update dates  
✅ **Keyboard Shortcuts** - Press `Ctrl+K` to search  

### Custom Enhancements

✅ **Logo Animation** - Heartbeat effect on hover  
✅ **Progress Bar** - Reading progress indicator  
✅ **Code Labels** - Language labels on code blocks  
✅ **External Links** - Auto-add icons to external links  
✅ **Smooth Scrolling** - Enhanced navigation experience  
✅ **Print Optimization** - Clean printing layout  

## 📊 Documentation Statistics

- **Total Pages**: 40+
- **API Reference Pages**: 30+
- **Code Examples**: 100+
- **Diagrams**: 10+
- **Supported Languages**: English (Spanish ready)

## 🤝 Contributing to Documentation

1. **Find the page** you want to edit in `docs/`
2. **Make changes** using Markdown
3. **Test locally** with `python build_docs.py --serve`
4. **Commit and push** your changes
5. **Documentation** auto-deploys on push to main

### Documentation Guidelines

- Use clear, concise language
- Include code examples
- Add type hints to all functions
- Follow existing structure
- Test locally before committing

## 📚 Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [mkdocstrings](https://mkdocstrings.github.io/)
- [Markdown Guide](https://www.markdownguide.org/)

## 🐛 Troubleshooting

### Documentation Not Building

```powershell
# Clear cache and rebuild
Remove-Item -Recurse -Force site
mkdocs build --clean
```

### Import Errors

Make sure you're in the Tools directory:

```powershell
cd Tools
$env:PYTHONPATH = (Get-Location).Path
mkdocs build
```

### Port Already in Use

```powershell
# Use different port
python build_docs.py --serve --port 8001
```

### Changes Not Showing

```powershell
# Hard refresh in browser
Ctrl + Shift + R (Windows)
Cmd + Shift + R (Mac)
```

## 📄 License

This documentation is part of the Mortality AMI Predictor project and follows the MIT License.

---

<div align="center">
  <p><strong>Built with ❤️ using MkDocs Material</strong></p>
  <p>
    <a href="http://localhost:8000">View Docs</a> •
    <a href="https://github.com/Pol4720/mortality-ami-predictor">GitHub</a>
  </p>
</div>
