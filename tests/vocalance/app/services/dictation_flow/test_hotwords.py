import pytest

from vocalance.app.services.dictation_flow.hotwords import HotwordValidationError, parse_hotword_text
from vocalance.app.services.storage.storage_models import DictationHotwordData


def test_parse_hotword_text_keeps_case_and_drops_duplicates():
    assert parse_hotword_text("API\n\n  Api  \nAPI\n") == ["API", "Api"]


@pytest.mark.parametrize(
    "text",
    [
        "foo bar",
        "A/B",
        ":score",
        "#threshold",
        "@phrase",
        "x" * 81,
    ],
)
def test_parse_hotword_text_rejects_unsafe_lines(text):
    with pytest.raises(HotwordValidationError):
        parse_hotword_text(text)


def test_parse_hotword_text_rejects_more_than_500():
    text = "\n".join(f"w{i}" for i in range(501))
    with pytest.raises(HotwordValidationError):
        parse_hotword_text(text)


def test_hotword_storage_model_rejects_internal_space():
    with pytest.raises(ValueError):
        DictationHotwordData(hotwords=["foo bar"])


async def test_save_text_persists_normalized_list(dictation_hotword_service):
    saved = await dictation_hotword_service.save_text("Zynthara\nZynthara\nAPI")

    assert saved is True
    assert dictation_hotword_service.get_hotwords() == ["Zynthara", "API"]
    dictation_hotword_service.storage.write.assert_called_once()


async def test_save_text_keeps_storage_when_a_line_has_a_space(dictation_hotword_service):
    await dictation_hotword_service.save_text("API")
    dictation_hotword_service.storage.write.reset_mock()

    saved = await dictation_hotword_service.save_text("API\nfoo bar")

    assert saved is False
    assert dictation_hotword_service.get_hotwords() == ["API"]
    dictation_hotword_service.storage.write.assert_not_called()


async def test_stream_hotwords_joins_saved_words(dictation_hotword_service):
    dictation_hotword_service.hotwords = ["API", "hyperkalemia"]

    assert dictation_hotword_service.stream_hotwords() == "API/hyperkalemia"
