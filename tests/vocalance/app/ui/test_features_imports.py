def test_feature_views_import():
    from vocalance.app.ui.features.commands.view import QtCommandsView
    from vocalance.app.ui.features.dictation.view import QtDictationView
    from vocalance.app.ui.features.marks.view import QtMarksView
    from vocalance.app.ui.features.overlays.mark_overlay import QtMarkView
    from vocalance.app.ui.features.settings.view import QtSettingsView
    from vocalance.app.ui.features.sounds.view import QtSoundsView

    assert QtCommandsView is not None
    assert QtDictationView is not None
    assert QtMarksView is not None
    assert QtMarkView is not None
    assert QtSettingsView is not None
    assert QtSoundsView is not None


def test_ui_registry_import():
    from vocalance.app.ui.application.ui_registry import UiRegistry

    assert UiRegistry is not None
