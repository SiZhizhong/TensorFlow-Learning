import tensorflow as tf

#config for Model
class config:
    def __init__(self,training_rate=0.1):
        self.training_rate=training_rate


class MnistModel:
    def __init__(self):
        raise NotImplementedError

    def read_in_data(self):
        raise NotImplementedError

    def create_placeholder(self):
        raise  NotImplementedError

    def create_traning_variable(self):
        raise  NotImplementedError

