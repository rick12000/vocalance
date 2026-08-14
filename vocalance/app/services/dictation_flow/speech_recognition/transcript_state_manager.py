from __future__ import annotations

DEFAULT_STABILITY_WINDOW: int = 2
DEFAULT_PROVISIONAL_WORDS: int = 4


class TranscriptStateManager:

    def __init__(self, stability_window: int = DEFAULT_STABILITY_WINDOW, provisional_words: int = DEFAULT_PROVISIONAL_WORDS) -> None:
        self.stability_window = max(1, stability_window)
        self.provisional_words = max(0, provisional_words)
        self.prev_words: list[str] = []
        self.stability_counts: list[int] = []
        self.committed_word_count: int = 0
        self.total_committed_text: str = ""
        self.last_sent_committed_len: int = 0

    def update(self, new_hypothesis: str) -> tuple[str, str]:
        new_words = new_hypothesis.split() if new_hypothesis.strip() else []
        n = len(new_words)

        new_counts: list[int] = []
        for i, word in enumerate(new_words):
            if i < len(self.prev_words) and word == self.prev_words[i]:
                prev = self.stability_counts[i] if i < len(self.stability_counts) else 0
                new_counts.append(min(prev + 1, self.stability_window + 1))
            else:
                new_counts.append(1)

        self.stability_counts = new_counts
        self.prev_words = new_words

        provisional_boundary = max(0, n - self.provisional_words)
        new_committed_count = 0
        for i in range(provisional_boundary):
            if self.stability_counts[i] >= self.stability_window:
                new_committed_count = i + 1
            else:
                break

        self.committed_word_count = max(self.committed_word_count, new_committed_count)

        committed_words = new_words[: self.committed_word_count] if new_words else []
        self.total_committed_text = " ".join(committed_words)

        committed_delta = self.extract_delta()

        provisional_words_tail = new_words[self.committed_word_count :]
        provisional_text = " ".join(provisional_words_tail)

        return committed_delta, provisional_text

    def finalize(self, final_hypothesis: str) -> str:
        new_words = final_hypothesis.split() if final_hypothesis.strip() else []
        self.committed_word_count = len(new_words)
        self.total_committed_text = " ".join(new_words)
        return self.extract_delta()

    def reset(self) -> None:
        self.prev_words = []
        self.stability_counts = []
        self.committed_word_count = 0
        self.total_committed_text = ""
        self.last_sent_committed_len = 0

    def extract_delta(self) -> str:
        full = self.total_committed_text
        already_sent = self.last_sent_committed_len
        if len(full) <= already_sent:
            return ""
        delta = full[already_sent:]
        if already_sent > 0 and delta and not delta.startswith(" "):
            delta = " " + delta
        self.last_sent_committed_len = len(full)
        return delta.strip()
