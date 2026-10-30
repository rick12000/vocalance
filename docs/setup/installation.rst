Installation
============

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/rick12000/vocalance.git
      cd vocalance

2. Install [uv](https://github.com/astral-sh/uv) and ensure Python **3.13** is available to it.

3. Install from the lockfile:

   **Base install** — no LLM / smart-dictation features:

   .. code-block:: bash

      uv sync --frozen

   **With LLM features** — smart dictation and text-amend (requires
   `Microsoft C++ Build Tools`_ with "Desktop development with C++" workload):

   .. code-block:: bash

      uv sync --frozen --extra llm

   **Development** (tests, docs, prek hooks):

   .. code-block:: bash

      uv sync --frozen --extra dev

4. Run:

   .. code-block:: bash

      uv run python vocalance.py

.. _Microsoft C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
