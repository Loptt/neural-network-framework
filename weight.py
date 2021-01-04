
class Weight():

    def __init__(self, value=0.0, previous_delta=0.0):
        self.value = value
        self.previous_val = value
        self.previous_delta = previous_delta

    def add_delta(self, delta_w):
        self.previous_val = self.value
        self.value -= delta_w
        self.previous_delta = delta_w

    def __str__(self):
        return "{:.2f}".format(self.value)
