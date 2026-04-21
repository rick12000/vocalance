from concurrent.futures import ThreadPoolExecutor

# A centralized, shared ThreadPoolExecutor dedicated entirely to OS-level input injection
# (e.g., pyautogui). Using max_workers=1 guarantees that mouse clicks and keystrokes
# are executed strictly sequentially in the exact order they were requested, preventing
# OS-level race conditions and ensuring that these blocking calls never freeze the
# main asyncio/Qt event loop.
shared_input_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="PyAutoGUI_Input_Thread")
