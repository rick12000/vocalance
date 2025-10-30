Installation
============

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/rick12000/vocalance.git

2. Create a Python 3.11 environment using your preferred environment manager.
3. Activate the environment.

4. Navigate to the repository directory:

   .. code-block:: bash

      cd vocalance

5. Install Vocalance as a package (on first set up, this may take a while due to `llama-cpp-python` dependancy):

   .. code-block:: bash

      pip install .

6. Run the application:

   .. code-block:: bash

      python vocalance.py

On first run, the application will download required assets (LLM model and tensorlfow YAMNet model), which may take several minutes depending on your internet connection.

You're now ready to use Vocalance or QA any changes you make to the codebase.
For detailed information about the codebase and development setup, see the :doc:`../developer/introduction` section.
