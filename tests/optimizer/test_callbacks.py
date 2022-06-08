from tensorwaves.optimizer.callbacks import CallbackList


class TestCallbackList:
    def test_eq(self):
        assert CallbackList([]) == CallbackList([])
