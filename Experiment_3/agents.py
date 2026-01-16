
class Agent:
    def __init__(self, id, prior, neighbors, age):
        self.id = id
        self.prior = prior
        self.neighbors = neighbors
        self.age = 1
        self.comm_succ = 1

