Dependencies and Licenses
=========================

Overview
--------

This page provides an overview of all dependencies required by SHARPIE and their 
licensing terms. Understanding these licenses is important for researchers who 
wish to use SHARPIE, especially if you plan to distribute modified versions or 
incorporate SHARPIE into commercial products.

SHARPIE License
---------------

SHARPIE is released under the `Apache License 2.0`_. This is a permissive 
open-source license that allows:

- Commercial use
- Modification
- Distribution
- Patent use
- Private use

.. _Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0

Python (PIP) Dependencies
-------------------------

.. include:: ../../DEPENDENCIES_PIP.md
   :parser: myst_parser

System Dependencies
-------------------

The following system-level dependencies are required:

Redis Server
   **License**: BSD-3-Clause
   
   **Purpose**: Required by ``channels_redis`` for WebSocket communication between Django channels layers.
   
   Installation on Ubuntu/Debian::
   
      sudo apt-get install redis-server
   
   Installation on macOS (Homebrew)::
   
      brew install redis

Graphviz
   **License**: Common Public License (EPL-1.0)
   
   **Purpose**: Required by ``pygraphviz`` for generating data model diagrams via ``django-extensions``.
   
   Installation on Ubuntu/Debian::
   
      sudo apt-get install graphviz libgraphviz-dev
   
   Installation on macOS (Homebrew)::
   
      brew install graphviz

License Compatibility Notes
---------------------------

Permissive Licenses (MIT, BSD, Apache)
   These licenses are fully compatible with Apache 2.0 and impose minimal 
   requirements. You can freely use, modify, and distribute SHARPIE with 
   dependencies under these licenses.

LGPL (Lesser General Public License)
   Can be used as a library dependency without affecting your project's 
   license. You can link to LGPL libraries without releasing your code 
   under LGPL.

GPL/AGPL (General Public License / Affero GPL)
   If GPL-licensed dependencies are included and distributed, the combined 
   work would need to comply with GPL terms. This could affect distribution 
   rights. The license report flags these cases for awareness. Note that 
   using GPL software as a service (without distribution) typically does 
   not trigger GPL requirements for your own code.