Installation
============

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/rick12000/vocalance.git
      cd vocalance

2. Create a Python **3.13.9** environment and activate it.

3. Install:

   .. code-block:: bash

      pip install .

4. Run:

   .. code-block:: bash

      python vocalance.py

On first launch, Vocalance downloads two large assets: the **YAMNet** sound
model and the **LLM bundle** used for smart dictation. Allow several minutes.
The Vosk and Moonshine speech models are bundled and require no download.
