# Documentation Style Guide

This guide defines the standards for documentation in the ezpz project.

## Python Docstrings

We follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings.

### Format

```python
def function_name(arg1: type, arg2: type = default) -> return_type:
    """Brief description of the function.
    
    Longer description of what the function does, if needed. This can span
    multiple lines and should explain the purpose and behavior of the function.
    
    Args:
        arg1 (type): Description of arg1.
        arg2 (type, optional): Description of arg2. Defaults to default.
        
    Returns:
        return_type: Description of what is returned.
        
    Raises:
        ExceptionType: Description of when this exception is raised.
        
    Example:
        >>> function_name(1, 2)
        3
        >>> function_name("hello", "world")
        'hello world'
    """
```

### Sections

1. **Brief Description**: One-line summary of what the function does
2. **Detailed Description**: Longer explanation if needed (optional)
3. **Args**: Document each parameter with its type and description
4. **Returns**: Document the return value with its type and description
5. **Raises**: Document any exceptions that might be raised (if applicable)
6. **Example**: Provide usage examples (if helpful)

### Type Annotations

Always use type annotations for function parameters and return values. Use `typing` module for complex types:

```python
from typing import Optional, Union, List, Dict

def example_function(
    name: str, 
    age: int, 
    emails: Optional[List[str]] = None,
    metadata: Dict[str, Union[str, int]] = None
) -> bool:
    """Example function with proper type annotations.
    
    Args:
        name (str): Person's name.
        age (int): Person's age.
        emails (List[str], optional): List of email addresses. Defaults to None.
        metadata (Dict[str, Union[str, int]], optional): Additional metadata.
            Defaults to None.
            
    Returns:
        bool: True if successful, False otherwise.
    """
    # implementation
```

## Shell Script Documentation

Document shell scripts using header comments:

```bash
#!/bin/bash
# @file script_name.sh
# @brief Brief description of what the script does
# @description
#     Longer description of the script's functionality.
#     This can span multiple lines.
#
# @author Your Name
# @date 2023-12-01
# @version 1.0.0

# Function documentation
# @description Brief description of what the function does
# @param $1 Description of first parameter
# @param $2 Description of second parameter
# @return Description of return value
function_name() {
    # implementation
}
```

## Markdown Documentation

For documentation files (README.md, docs/, etc.):

### Headers

Use proper header hierarchy:
```markdown
# Main Title (H1)
## Section (H2)
### Subsection (H3)
#### Sub-subsection (H4)
```

### Code Blocks

Always specify the language for syntax highlighting:
```markdown
```python
def example():
    pass
```

```bash
echo "Hello World"
```
```

### Lists

Use consistent formatting:
```markdown
- First item
- Second item
  - Indented item
  - Another indented item

1. Numbered item
2. Second numbered item
```

## API Documentation

API documentation is generated using [mkdocstrings](https://mkdocstrings.github.io/). Ensure all public functions, classes, and modules have proper docstrings.

### Module Documentation

Each module should start with a module docstring:

```python
"""Brief description of the module.

Longer description of what this module provides and how to use it.

Example:
    >>> import module_name
    >>> result = module_name.function()
"""
```

### Class Documentation

Document classes with their purpose, attributes, and methods:

```python
class ExampleClass:
    """Brief description of the class.
    
    Longer description of the class functionality.
    
    Attributes:
        attribute_name (type): Description of the attribute.
    """
    
    def __init__(self, param: str):
        """Initialize the class.
        
        Args:
            param (str): Description of the parameter.
        """
        self.attribute_name = param
```

## Consistency

- Use American English spelling
- Use present tense ("Returns" not "Returned")
- Use active voice when possible
- Keep line lengths reasonable (generally < 100 characters)
- Be concise but clear
- Use proper punctuation and grammar

## Tools

We use the following tools to maintain documentation quality:

- **pydocstyle**: Validates docstring conventions
- **mypy**: Checks type annotations
- **mkdocs**: Builds documentation site
- **mkdocstrings**: Generates API documentation from docstrings

Run these tools regularly to ensure documentation quality.