Installation
============

Vocalance can be installed in several ways depending on your needs and technical comfort level.

System Requirements
-------------------

**Operating System**
   - Windows 10/11 (macOS and Linux support planned)

**Hardware Requirements**
   - **Memory**: 2GB RAM minimum
   - **Microphone**: A quality headset or microphone is strongly recommended for best recognition accuracy
   - **Storage**: ~2GB free space for models and application data

**Software Requirements**
   - Python 3.9 or higher (for source installation)

Installation Options
--------------------

Option 1: Install from Source (Recommended for Developers)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This method allows you to run Vocalance directly from the source code and make modifications if needed.

1. **Clone the repository:**

   .. code-block:: bash

      git clone https://github.com/rick12000/vocalance.git
      cd vocalance

2. **Create a Python environment:**

   We recommend using conda or venv to create an isolated environment:

   .. code-block:: bash

      # Using conda (recommended)
      conda create -n vocalance python=3.11
      conda activate vocalance

      # Or using venv
      python -m venv vocalance_env
      vocalance_env\Scripts\activate  # On Windows

3. **Install Vocalance:**

   .. code-block:: bash

      pip install .

4. **Run the application:**

   .. code-block:: bash

      python vocalance.py

   The application will download required models (speech recognition, AI models) on first run. This may take several minutes depending on your internet connection.

Option 2: Install as a Python Package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If Vocalance is published to PyPI in the future, you can install it directly:

.. code-block:: bash

   pip install vocalance

Dependencies
------------

Vocalance requires several dependencies for audio processing, speech recognition, and GUI functionality. These are automatically installed when using pip:

**Core Dependencies:**
   - **Audio Processing**: vosk, sounddevice, librosa, soundfile, numpy
   - **Speech Recognition**: faster-whisper, llama-cpp-python
   - **Machine Learning**: scikit-learn, joblib, tensorflow-hub, tensorflow
   - **GUI Framework**: customtkinter, Pillow
   - **System Automation**: PyAutoGUI, pyperclip, psutil
   - **Configuration**: pydantic, PyYAML, appdirs
   - **Networking**: requests, tqdm, huggingface_hub

**Development Dependencies:**
   Additional tools for testing and documentation are available in ``requirements-dev.txt`` and can be installed with:

   .. code-block:: bash

      pip install -r requirements-dev.txt

Troubleshooting
---------------

**Common Issues:**

1. **Model Download Failures:**
   - Ensure you have a stable internet connection
   - Check firewall settings allow downloads from HuggingFace
   - Models are cached locally after first download

2. **Audio Device Issues:**
   - Verify your microphone is properly connected and selected as default
   - Check Windows privacy settings for microphone access
   - Test audio input in other applications

3. **Permission Errors:**
   - Run the command prompt as Administrator if needed
   - Ensure write permissions in the installation directory

4. **Python Version Issues:**
   - Vocalance requires Python 3.9+
   - Use ``python --version`` to check your Python version

For additional support, contact the development team at vocalance.contact@gmail.com.
