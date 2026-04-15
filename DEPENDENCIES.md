# SHARPIE Dependencies and Licenses

This document lists all dependencies required for SHARPIE and their licenses.

## Python (PIP) Dependencies

See [DEPENDENCIES_PIP.md](DEPENDENCIES_PIP.md) for the list of Python packages and their licenses.

## Updating This Document

To regenerate the Python dependencies license report after changing requirements:

```bash
python scripts/generate_dependencies.py
git add DEPENDENCIES_PIP.md
git commit -m "Update dependencies license report"
```

## System Dependencies

| Dependency | License | Purpose | Installation |
|------------|---------|---------|--------------|
| Redis Server | BSD-3-Clause | WebSocket channels backend | `sudo apt-get install redis-server` |
| Graphviz | EPL-1.0 | Data model diagram generation | `sudo apt-get install graphviz libgraphviz-dev` |

### Redis Server

- **License**: BSD-3-Clause
- **Purpose**: Required by `channels_redis` for WebSocket communication between Django channels layers
- **Installation on Ubuntu/Debian**: `sudo apt-get install redis-server`
- **Installation on macOS**: `brew install redis`

### Graphviz

- **License**: Common Public License (EPL-1.0)
- **Purpose**: Required by `pygraphviz` for generating data model diagrams via `django-extensions`
- **Installation on Ubuntu/Debian**: `sudo apt-get install graphviz libgraphviz-dev`
- **Installation on macOS**: `brew install graphviz`

## License Compatibility

SHARPIE is licensed under [Apache License 2.0](LICENSE).

### Compatible Licenses

The following license types are generally compatible with Apache 2.0:

- **Permissive licenses**: MIT, BSD-2-Clause, BSD-3-Clause, Apache-2.0, ISC
- **Weak copyleft**: LGPL-2.1, LGPL-3.0, MPL-2.0 (can be used as libraries)

### Notes on Strong Copyleft Licenses

If dependencies licensed under GPL-2.0, GPL-3.0, or AGPL-3.0 are included, the combined work would need to comply with those license terms. This could affect distribution rights.