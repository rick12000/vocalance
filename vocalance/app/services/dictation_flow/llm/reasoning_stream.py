class ReasoningTokenFilter:
    """Drop chain-of-thought spans so only the final answer is streamed.

    Models that think wrap that span in ``<think>`` ... ``</think>``. The tags
    may be split across streamed tokens. When the chat template already opened
    a think block in the prompt, generation starts inside that span and the
    filter waits for the closing tag before emitting anything.
    """

    def __init__(
        self,
        open_tag: str = "<think>",
        close_tag: str = "</think>",
        start_inside: bool = False,
    ) -> None:
        self.open_tag = open_tag
        self.close_tag = close_tag
        self.inside = start_inside
        self._buffer = ""
        self._drop_leading_whitespace = start_inside

    def push(self, token: str) -> str:
        """Return the visible portion of ``token``. Reasoning text is omitted."""
        if not token:
            return ""
        self._buffer += token
        visible: list[str] = []
        while self._buffer:
            if self.inside:
                close_at = self._buffer.find(self.close_tag)
                if close_at < 0:
                    self._buffer = _partial_marker_suffix(self._buffer, self.close_tag)
                    break
                self._buffer = self._buffer[close_at + len(self.close_tag) :]
                self.inside = False
                self._drop_leading_whitespace = True
                continue
            open_at = self._buffer.find(self.open_tag)
            if open_at < 0:
                held = _partial_marker_suffix(self._buffer, self.open_tag)
                emit = self._buffer[: len(self._buffer) - len(held)] if held else self._buffer
                self._buffer = held
                visible.append(self._take_visible(emit))
                break
            visible.append(self._take_visible(self._buffer[:open_at]))
            self._buffer = self._buffer[open_at + len(self.open_tag) :]
            self.inside = True
        return "".join(visible)

    def finish(self) -> str:
        """Flush text still held because it might have been a split tag."""
        if self.inside:
            self._buffer = ""
            return ""
        pending = self._buffer
        self._buffer = ""
        return self._take_visible(pending)

    def _take_visible(self, text: str) -> str:
        if not text:
            return ""
        if self._drop_leading_whitespace:
            stripped = text.lstrip()
            if not stripped:
                return ""
            self._drop_leading_whitespace = False
            return stripped
        return text


def _partial_marker_suffix(text: str, marker: str) -> str:
    """Return the trailing prefix of ``marker`` that ``text`` might be starting."""
    limit = min(len(text), len(marker) - 1)
    for size in range(limit, 0, -1):
        if marker.startswith(text[-size:]):
            return text[-size:]
    return ""
