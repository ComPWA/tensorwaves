from tensorwaves.optimizer.callbacks import CallbackList


def describe_CallbackList():
    def it_compares_equal_when_it_wraps_the_same_callbacks():
        assert CallbackList([]) == CallbackList([])
