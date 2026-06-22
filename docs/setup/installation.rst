Installation
============

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/rick12000/vocalance.git
      cd vocalance

2. Create a Python **3.13.9** environment and activate it.

3. Install (choose one):

   **Base install** — no LLM / smart-dictation features:

   .. code-block:: bash

      pip install .

   **With LLM features** — smart dictation and text-amend (requires
   `Microsoft C++ Build Tools`_ with "Desktop development with C++" workload):

   .. code-block:: bash

      pip install ".[llm]"

   When using UV with the lock file instead of pip:

   .. code-block:: bash

      uv sync               # base only
      uv sync --extra llm   # base + LLM features

4. Run:

   .. code-block:: bash

      python vocalance.py

On first launch, Vocalance downloads required assets: the **YAMNet** sound
model for custom sound recognition and, if LLM features are enabled, the
**LLM bundle** used for smart dictation. Allow several minutes for these
downloads. The Vosk and Moonshine speech models are bundled and require no
separate download.

.. note::

   LLM features (smart dictation, text-amend) are **optional**.  When not
   installed, the smart-dictation UI sections and their voice triggers are
   automatically hidden.  All other features work without any LLM dependency.

.. _Microsoft C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
