from __future__ import annotations

from typing import Any, Callable, Dict, List, NamedTuple

from vocalance.app.lifecycle.lifecycle import AppLifecycle


class ServiceSpec(NamedTuple):
    """Declarative description of one service in the construction graph.

    Spec order in a list defines both construction order and the LIFO teardown
    order: a list ``[a, b, c]`` builds as ``a -> b -> c`` and tears down as
    ``c -> b -> a``.

    Attributes:
        name: Key under which the constructed service is exposed in the build
            context. Subsequent specs may depend on this key.
        factory: Callable receiving the partially-built context and returning
            the constructed service. Heavy initialization should be deferred to
            the service's ``initialize`` coroutine; otherwise wrap blocking work
            in ``run_blocking`` so the calling thread is not stalled.
    """

    name: str
    factory: Callable[[Dict[str, Any]], Any]


def build_services(specs: List[ServiceSpec], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Construct each service in ``specs`` order, exposing each under its name in ``ctx``.

    Args:
        specs: Ordered service specifications. Each factory may consume any
            previously-built service or base dependency from ``ctx``.
        ctx: Initial context (base dependencies such as ``event_bus``,
            ``config``, ``gui_loop``). Mutated in place.

    Returns:
        The same ``ctx`` dict with all built services added.
    """
    for spec in specs:
        ctx[spec.name] = spec.factory(ctx)
    return ctx


def register_services_for_teardown(
    specs: List[ServiceSpec],
    ctx: Dict[str, Any],
    lifecycle: AppLifecycle,
) -> None:
    """Register every constructed service with ``lifecycle`` in spec order.

    LIFO teardown semantics on ``AppLifecycle`` cause services to shut down in
    reverse declaration order.
    """
    for spec in specs:
        lifecycle.register_resource(ctx[spec.name])
