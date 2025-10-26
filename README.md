<div style="width:100%; display:flex; justify-content:center;">
  <img src="vocalance/app/assets/repo/banner_github.png" alt="Vocalance Logo" style="width:100%; max-width:1000px; height:auto;"/>
</div>

<!-- TODO: update links: -->
<div align="center">
  <a href="https://vocalance.readthedocs.io/en/latest/getting_started.html">Website</a> |
  <a href="https://vocalance.readthedocs.io/en/latest/basic_usage.html">Demo</a> |
  <a href="https://vocalance.readthedocs.io/en/latest/api_reference.html">User Guide</a>
</div>


## 💡 Overview

Vocalance offers hands free control of your computer, enabling you to switch tabs, move on screen, dictate anywhere and much more!

## 🚀 Getting Started



## 💻 Build from Source

If you prefer to run Vocalance directly from the source code in this repository, rather than running it as an executable, follow the instructions below.

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rick12000/vocalance.git
   ```

2. **Create a Python 3.11 environment using your environment manager of choice. Then with that environment activated:**

    - **Go to repository directory:**
      ```bash
      cd vocalance
      ```

    - **Install Vocalance as a package locally:**
      ```bash
      pip install .
      ```

    - **Run the application:**
        ```bash
        python vocalance.py
        ```

The application will start up and download any required models (like speech recognition models) on first run. This may take several minutes depending on your internet connection.

On follow up runs, skip the `pip install .` step.

## 🔧 System Requirements

- **Operating System**: Windows 10/11 (macOS and Linux support planned)
- **Memory**: 2GB RAM
- **Hardware**: It is **strongly** recommended to purchase a reasonably good headset or microphone to improve Vocalance outputs and recognition, but it will still work without this.

## 🤝 Contributing

Reach out at vocalance.contact@gmail.com with title **"Contribution"** if:

- You have software engineering experience and have feedback on how the architecture of the application could be improved.
- You want to add an original or pre-approved feature.

For now, contributions will be handled on an ad-hoc basis, but in future contribution guidelines will be set up depending on the number of contributors.

## 📈 Upcoming Features

The following features are planned additions to Vocalance, with some in early development and others under consideration:

*   **Eye Tracking for Cursor Control:** This feature is planned to enable cursor control via eye movements.
    *   **Gaze Tracking Accuracy:** Merge gaze tracking with historical screen click data and screen contents to improve accuracy, aiming for good performance even with webcam tracking.
    *   **Zoom Option:** Add a zoom option to better direct gaze on screen contents.

*   **Context-Aware Commands:** Implement context bucketing for commands, allowing the same command phrase (e.g., "previous") to map to different hotkeys depending on the active application (e.g., VSCode vs. Chrome). This aims to avoid disambiguation phrases.

*   **LLM-Powered Text Refactoring:** Ability to select any text and reformat it via an LLM by speaking a prompt.

*   **Improved Text Editing & Navigation:** Further enhancements to text editing and text navigation tools.

*   **Enhanced Predictive Features:** Improve predictive capabilities based on window contents, recent context, gaze patterns, and more.
    *   *Privacy Note:* Any feature requiring local storage of potentially sensitive data (e.g., screenshots, window contents) will be deployed as an opt-in feature and disabled by default.
