import vocalance.app.ui.components.layout_composites as lcm
import vocalance.app.ui.components.layout_core as lc
from vocalance.app.ui.components import layouts


def test_layout_core_has_scrollable():
    assert hasattr(lc, "ScrollableContainer")
    assert hasattr(lc, "TransparentWidget")


def test_layout_composites_has_two_column():
    assert hasattr(lcm, "TwoColumnLayout")
    assert hasattr(lcm, "ListForm")


def test_layouts_shim_reexports():
    for name in layouts.__all__:
        assert hasattr(layouts, name), name
