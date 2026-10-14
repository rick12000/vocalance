from vocalance.app.config.app_config import DEFAULT_LLM_MODEL_ID, LLMConfig, local_llm_allowlist


def test_allowlist_defaults_to_qwen35_2b_with_thinking_off():
    allowlist = local_llm_allowlist()
    default = allowlist.artifact_for(DEFAULT_LLM_MODEL_ID)

    assert [artifact.id for artifact in allowlist.artifacts] == [
        "qwen3.5-2b-q5km",
        "qwen3-4b-instruct-2507-q5km",
        "granite-4.2-3b-q5km",
        "qwen3.5-4b-q5km",
    ]
    assert default is not None
    assert default.id == "qwen3.5-2b-q5km"
    assert default.chat_template_kwargs() == {"enable_thinking": False}
    assert default.stream_starts_in_reasoning is False
    assert default.active_sampling().temperature == 1.0
    assert default.active_sampling().presence_penalty == 2.0


def test_thinking_capable_models_can_switch_to_their_thinking_profile():
    granite = local_llm_allowlist().artifact_for("granite-4.2-3b-q5km")
    qwen4b = local_llm_allowlist().artifact_for("qwen3.5-4b-q5km")
    assert granite is not None and qwen4b is not None

    thinking_granite = granite.model_copy(update={"disable_thinking": False})
    thinking_qwen = qwen4b.model_copy(update={"disable_thinking": False})

    assert thinking_granite.chat_template_kwargs() == {"enable_thinking": True}
    assert thinking_granite.stream_starts_in_reasoning is True
    assert thinking_granite.active_sampling().temperature == 1.0
    assert thinking_granite.active_sampling().top_p == 0.95
    assert thinking_qwen.active_sampling().temperature == 1.0
    assert thinking_qwen.active_sampling().top_p == 0.95
    assert qwen4b.active_sampling().temperature == 0.7
    assert qwen4b.active_sampling().top_p == 0.8


def test_instruct_2507_has_no_thinking_switch():
    instruct = local_llm_allowlist().artifact_for("qwen3-4b-instruct-2507-q5km")

    assert instruct is not None
    assert instruct.supports_thinking is False
    assert instruct.chat_template_kwargs() == {}
    assert instruct.stream_starts_in_reasoning is False
    assert instruct.active_sampling().temperature == 0.7
    assert instruct.active_sampling().top_p == 0.8
    assert instruct.active_sampling().top_k == 20


def test_retired_model_id_falls_back_to_the_current_default():
    config = LLMConfig(selected_model_id="qwen2.5-1.5b-q5km")

    assert config.selected_model_id == DEFAULT_LLM_MODEL_ID
