class InputCell:
    def __init__(self, initial_value):
        self.value = initial_value

class ComputeCell:
    def __init__(self, inputs, compute_function):
        self.value = None
        self Inputs = inputs
        self.compute_function = compute_function
        self.callback_list = []

    def add_callback(self, callback):
        self.callback_list.append(callback)

    def remove_callback(self, callback):
        self.callback_list.remove(callback)

    def on_change(self, prev_value, new_value):
        self.callback_list.append(lambda: None)