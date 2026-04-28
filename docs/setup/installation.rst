Installation
============

Vocalance can be installed two ways: as an end user running the released
application, or as a developer working on the codebase.

End-user setup
--------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/rick12000/vocalance.git
      cd vocalance

2. Create a Python **3.13.9** environment and activate it. The exact patch
   version matters because the C-extension stack (``llama-cpp-python``,
   ``tensorflow-cpu``, ``moonshine-voice``) is sensitive to ABI drift.
   ``pyproject.toml`` pins ``requires-python = "==3.13.9"``.

3. Install:

   .. code-block:: bash

      pip install .

   The first install is slow because ``llama-cpp-python`` is built from
   source on most platforms.

4. Run:

   .. code-block:: bash

      python vocalance.py

On first launch the application downloads two large assets it cannot ship
in the wheel: the **YAMNet** model used for sound recognition, and the
**LLM bundle** used by smart and amend dictation. Allow several minutes on
a slow network. The Vosk and Moonshine speech models are bundled and
do not need a download.

Developer setup
---------------

If you intend to modify the codebase, an editable install is more
convenient:

.. code-block:: bash

   conda create -n vocalance_env_dev python=3.13.9
   conda activate vocalance_env_dev
   uv pip install -e ".[dev]"

``uv`` is recommended for speed; plain ``pip install -e ".[dev]"`` works
identically. To run the test suite:

.. code-block:: bash

   pytest

To build the documentation locally:

.. code-block:: bash

   cd docs
   make.bat html        # Windows
   make html            # Linux/macOS

You're now ready to make changes. Continue with
:doc:`../developer/overview/introduction` for an architectural tour.
