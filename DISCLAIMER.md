# Disclaimer and Legal Notices

> This document is provided to give users clear, plain-language information about how Vocalance works, what it does on your machine, and the limitations of any guarantees. It supplements — and does not replace — the [GNU General Public License v3](https://github.com/rick12000/vocalance/blob/main/LICENSE.txt) under which Vocalance is distributed.

---

## 1. No Warranty

Vocalance is distributed **"AS IS"**, without warranty of any kind. To the maximum extent permitted by applicable law:

- There is **no guarantee** that Vocalance will work correctly, continuously, or without error on your specific hardware, Windows configuration, or with any particular application.
- There is **no guarantee** of fitness for any purpose, including medical, accessibility, or professional use.
- There is **no guarantee** of accuracy from any speech recognition, language model, or sound recognition component.

The complete formal warranty disclaimer is in [LICENSE.txt §§15–16](https://github.com/rick12000/vocalance/blob/main/LICENSE.txt).

---

## 2. Limitation of Liability

The authors and contributors of Vocalance shall not be liable for any damages arising from use of this software, including but not limited to:

- Unintended keystrokes, mouse clicks, or other automated OS actions resulting from speech recognition errors or AI misinterpretation.
- Loss of data, files, or work caused directly or indirectly by Vocalance's automation output.
- System instability, resource exhaustion, or conflicts with other software.
- Interrupted or failed model downloads leaving partial files on disk.
- Any harm arising from reliance on dictation accuracy for critical tasks.

---

## 3. Microphone Access

Vocalance requires **continuous access to your system microphone** to function. By running Vocalance, you acknowledge:

- Audio is captured from your default microphone in real time using PortAudio via the `sounddevice` library.
- Audio is processed **entirely on your local machine**. At no point is audio data, speech transcripts, or any derivative of your voice transmitted to any external server, cloud service, or third party by Vocalance.
- Audio is not written to disk. It exists only in memory during processing and is discarded once the relevant recognition pipeline has consumed it.
- Because Vocalance listens continuously while running, **it is your responsibility** to ensure that use of Vocalance complies with any applicable recording consent laws in your jurisdiction (e.g. laws requiring all-party consent before recording conversations).

---

## 4. OS-Level Automation

Vocalance controls your keyboard and mouse programmatically based on voice commands and AI outputs. You acknowledge:

- Speech recognition is **imperfect**. Misrecognised commands can and will occasionally cause unintended keypresses, mouse clicks, or text input in any focused application.
- **Vocalance is not suitable for use in safety-critical contexts** where unintended input could cause harm (e.g. operating industrial equipment, medical devices, or critical infrastructure).
- You should save open documents frequently and keep the ability to manually override or dismiss Vocalance's output at all times.

---

## 5. AI-Powered Features

Vocalance uses multiple AI and machine learning models. All AI processing runs **locally on your machine** — no data is sent to any AI service or cloud provider.

| Feature | AI Technology |
|---|---|
| Voice command recognition | Vosk (small English ASR model, bundled) |
| Dictation (speech-to-text) | Moonshine Voice (streaming STT, downloaded on first run) |
| Voice activity detection | Silero VAD (via Moonshine, downloaded on first run) |
| Environmental sound recognition | YAMNet + k-NN classifier (bundled) |
| Dictation post-processing / agentic | Large language model via llama-cpp-python (downloaded on first run) |

The following features do **not** use AI:

- Storage reads/writes.
- Keyboard and mouse automation execution.
- The settings UI, marks, commands, and alias management panels.
- Grid overlay rendering and navigation.

---

## 6. What Is Downloaded and When

Vocalance downloads AI model files on first launch only, to your local machine. **No user data, voice, or activity is ever uploaded.** The only outbound network activity Vocalance initiates is the download of these model files.

### 6.1 Moonshine Speech-to-Text Model

- **When:** Automatically on first launch if not already present.
- **Source:** Downloaded by the `moonshine-voice` Python package from its model distribution endpoint.
- **Stored at:** A system cache directory managed by the `moonshine-voice` package.
- **License:** Apache 2.0.

### 6.2 Language Model (LLM)

- **When:** Automatically on first launch if not already present. A download progress indicator is shown during startup.
- **Source:** Downloaded from [Hugging Face Hub](https://huggingface.co).
- **Default model:** Qwen 2.5 1.5B Instruct (GGUF format), licensed under Apache 2.0.
- **Stored at:**
  ```
  %APPDATA%\Vocalance\llm_models\
  ```
  On most Windows installations this resolves to:
  ```
  C:\Users\<YourUsername>\AppData\Roaming\Vocalance\llm_models\
  ```
- Additional language models are available for download through the settings UI. The user must explicitly click download for any additional model to be fetched. All additional models are also downloaded from Hugging Face.
- **Note on Hugging Face download statistics:** Hugging Face Hub records download counts for hosted model files. When Vocalance downloads a model on your behalf, Hugging Face registers one download event. No personal information from Vocalance is transmitted; this is solely Hugging Face's own infrastructure telemetry. See [Hugging Face's Privacy Policy](https://huggingface.co/privacy) for details.

---

## 7. What Is Stored on Your Machine

Vocalance writes user data exclusively to your local machine. **Nothing in this section is transmitted to any network.** All files are standard JSON and can be read, edited, or deleted with any text editor or file manager.

### 7.1 User Data Root Directory

All Vocalance user data is stored under:

```
%APPDATA%\Vocalance\
```

On most Windows installations this resolves to:

```
C:\Users\<YourUsername>\AppData\Roaming\Vocalance\
```

To navigate there directly: open **File Explorer**, click the address bar, and paste `%APPDATA%\Vocalance`.

### 7.2 File Inventory

| File | Location (relative to data root) | Contents |
|---|---|---|
| `marks.json` | `marks\marks.json` | Named screen position coordinates you have saved ("marks"). |
| `click_history.json` | `click_tracker\click_history.json` | History of grid overlay click events: x/y coordinates, timestamps, and cell identifiers. Used to improve grid click prediction. |
| `app_user_config.json` | `settings\app_user_config.json` | Your settings overrides (microphone, UI preferences, etc.). |
| `custom_commands.json` | `settings\custom_commands.json` | Custom voice commands you have defined, and any default command phrase overrides. |
| `settings.yaml` | `settings\settings.yaml` | Optional advanced configuration overrides. Only exists if you have manually created it. |
| `agentic_prompts.json` | `dictation\agentic_prompts.json` | System prompts you have configured for the LLM-assisted dictation feature. |
| `aliases.json` | `dictation\aliases.json` | Dictation text substitution shortcuts you have defined. |
| `sound_mappings.json` | `sound_models\sound_mappings.json` | Your custom sound-to-action mappings. |
| Sound samples | `sound_samples\` | Audio sample files used for custom sound recognition. May include user-recorded samples. |
| LLM model file(s) | `llm_models\` | Downloaded language model file(s) in GGUF format. Large files. Re-downloaded from Hugging Face if deleted. |

### 7.3 Log Files

By default, **logging is completely disabled** in the shipped application. When disabled, no log output is written anywhere — not to disk, not to the console.

If a developer or advanced user explicitly enables logging by modifying the `enable_logs` field in the application configuration, log files will be written to:

```
%APPDATA%\Vocalance\logs\<timestamp>\app.log
```

This applies only when `enable_logs` is set to `true`, which requires direct modification of the source code or configuration file. This setting is `false` by default and will not be active in a standard installation.

### 7.4 What Is NOT Collected or Stored

Vocalance does **not** collect, store, or transmit any of the following:

- Your voice or audio recordings.
- The text of anything you dictate.
- Screen contents or screenshots.
- Information about which applications you use.
- Crash reports or diagnostic data.
- Any form of usage analytics or telemetry.
- Your name, email address, or any personally identifying information.

---

## 8. Third-Party Software

Vocalance is built on and integrates with a number of third-party open-source libraries and models. Full licence texts for all dependencies are provided in [NOTICES/NOTICE.txt](https://github.com/rick12000/vocalance/blob/main/NOTICES/NOTICE.txt) and the individual files under [NOTICES/PYPI_LICENSES/](https://github.com/rick12000/vocalance/blob/main/NOTICES/PYPI_LICENSES/).

Key third-party components:

| Component | Purpose | Licence |
|---|---|---|
| Vosk | Command speech recognition | Apache 2.0 |
| Moonshine Voice | Dictation speech-to-text | Apache 2.0 |
| YAMNet (TensorFlow Hub) | Environmental sound classification | Apache 2.0 |
| Qwen 2.5 1.5B (GGUF) | Default LLM for dictation post-processing | Apache 2.0 |
| llama-cpp-python | LLM inference runtime | MIT |
| TensorFlow CPU | ML inference framework | Apache 2.0 |
| PySide6 | Desktop GUI framework | LGPL v3 |
| pyautogui | Keyboard/mouse automation | BSD 3-Clause |
| sounddevice | Microphone audio capture | MIT |
| huggingface-hub | Model download client | Apache 2.0 |

---

## 10. Contact

For questions, issues, or concerns regarding this software:

- **GitHub Issues:** [github.com/rick12000/vocalance](https://github.com/rick12000/vocalance)
- **Email:** vocalance.contact@gmail.com
