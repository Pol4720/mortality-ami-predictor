"""Generate API documentation pages from source code.

This script automatically generates mkdocs pages for all modules in the src/ directory.
Run this after adding new modules to update the documentation.
"""
import os
from pathlib import Path
from typing import List


def get_python_files(directory: Path) -> List[Path]:
    """Get all Python files in a directory."""
    return [f for f in directory.rglob("*.py") if f.name != "__init__.py" and not f.name.startswith("_")]


def get_module_path(file_path: Path, src_dir: Path) -> str:
    """Convert file path to Python module path."""
    rel_path = file_path.relative_to(src_dir.parent)
    module_path = str(rel_path).replace(os.sep, ".").replace(".py", "")
    return module_path


def generate_api_page(module_path: str, module_name: str) -> str:
    """Generate markdown content for an API page."""
    title = module_name.replace("_", " ").title()
    
    content = f"""# {title}

::: {module_path}
    options:
      show_source: true
      show_root_heading: true
      show_root_full_path: false
      heading_level: 2
      members_order: source
      group_by_category: true
"""
    return content


def main():
    """Generate API documentation pages."""
    # Paths
    tools_dir = Path(__file__).parent
    src_dir = tools_dir / "src"
    docs_api_dir = tools_dir / "docs" / "api"
    
    if not src_dir.exists():
        print(f"❌ Source directory not found: {src_dir}")
        return
    
    # Create api directory if it doesn't exist
    docs_api_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all Python modules
    python_files = get_python_files(src_dir)
    
    print(f"Found {len(python_files)} Python modules")
    
    # Generate pages for each module
    for py_file in python_files:
        # Get module info
        module_path = get_module_path(py_file, src_dir)
        module_name = py_file.stem
        
        # Create directory structure in docs
        rel_path = py_file.relative_to(src_dir)
        doc_dir = docs_api_dir / rel_path.parent
        doc_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate markdown file
        doc_file = doc_dir / f"{module_name}.md"
        content = generate_api_page(module_path, module_name)
        
        with open(doc_file, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"✅ Generated: {doc_file.relative_to(tools_dir)}")
    
    print(f"\n🎉 Generated {len(python_files)} API documentation pages!")
    print("\n💡 Next steps:")
    print("   1. Review generated pages in docs/api/")
    print("   2. Update mkdocs.yml navigation if needed")
    print("   3. Run 'python build_docs.py --serve' to preview")


if __name__ == "__main__":
    main()
