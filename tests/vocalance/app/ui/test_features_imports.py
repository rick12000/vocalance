def test_feature_packages_import():
    from vocalance.app.ui.features import commands, dictation, marks, overlays, settings, sounds

    assert hasattr(commands, "QtCommandsView")
    assert hasattr(dictation, "QtDictationView")
    assert hasattr(marks, "QtMarksView")
    assert hasattr(overlays, "QtMarkView")
    assert hasattr(settings, "QtSettingsView")
    assert hasattr(sounds, "QtSoundsView")


def test_ui_registry_import():
    from vocalance.app.ui.application.ui_registry import UiRegistry

    assert UiRegistry is not None
