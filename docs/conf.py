project = "SimDPS"
author = "Sean Turland"
copyright = "2026, Sean Turland"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "nbsphinx",
]
autodoc_member_order = "bysource"
autodoc_mock_imports = ["diffusers", "transformers", "safetensors"]
napoleon_numpy_docstring = True
nbsphinx_execute = "never"
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_title = "SimDPS"
