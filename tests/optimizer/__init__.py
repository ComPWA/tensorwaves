from __future__ import annotations

from enum import Enum
from typing import Any
from unittest.mock import MagicMock

from tensorwaves.optimizer.callbacks import Callback


class CallbackType(Enum):
    ON_OPTIMIZE_START = "on_optimize_start"
    ON_OPTIMIZE_END = "on_optimize_end"
    ON_ITERATION_END = "on_iteration_end"
    ON_FUNCTION_CALL_END = "on_function_call_end"


def assert_invocations(callback_stub: MagicMock):
    callback_args_order = [x[0] for x in callback_stub.call_args_list]
    assert callback_args_order[0][0] == CallbackType.ON_OPTIMIZE_START
    assert callback_args_order[-1][0] == CallbackType.ON_OPTIMIZE_END

    iterations = 1
    function_calls = 1
    for callback_args in callback_args_order[1:-1]:
        callback_type = callback_args[0]
        if callback_type == CallbackType.ON_FUNCTION_CALL_END:
            assert callback_args[1] == function_calls
            function_calls += 1
        elif callback_type == CallbackType.ON_ITERATION_END:
            assert callback_args[1] == iterations
            iterations += 1


class CallbackMock(Callback):
    def __init__(self, callback_stub: MagicMock):
        self.__callback_stub = callback_stub

    def on_optimize_start(self, logs: dict[str, Any] | None = None) -> None:
        self.__callback_stub(CallbackType.ON_OPTIMIZE_START, logs)

    def on_optimize_end(self, logs: dict[str, Any] | None = None) -> None:
        self.__callback_stub(CallbackType.ON_OPTIMIZE_END, logs)

    def on_iteration_end(
        self, iteration: int, logs: dict[str, Any] | None = None
    ) -> None:
        self.__callback_stub(CallbackType.ON_ITERATION_END, iteration, logs)

    def on_function_call_end(
        self, function_call: int, logs: dict[str, Any] | None = None
    ) -> None:
        self.__callback_stub(
            CallbackType.ON_FUNCTION_CALL_END, function_call, logs
        )
