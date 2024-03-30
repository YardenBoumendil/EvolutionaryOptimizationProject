import sys

from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator


class FitnessEvaluator(SimpleIndividualEvaluator):
    def __init__(self, f1, f2, g1, g2, c):
        super().__init__()
        self.f1 = f1
        self.f2 = f2
        self.g1 = g1
        self.g2 = g2
        self.c = c

    def evaluate_individual(self, individual):
        x1 = individual.cell_value(0)
        x2 = individual.cell_value(1)
        g_value = self.compute_g(x1, x2)
        if not g_value:
            return sys.maxsize

        return self.compute_f(x1, x2)

    def compute_f(self, x1, x2):
        if isinstance(self.f1,dict):
            return self.f1[x1] + self.f2[x2]

        return self.f1(x1) + self.f2(x2)


    # BOOL
    def compute_g(self, x1, x2):
        if isinstance(self.g1, dict):
            output = self.g1[x1] + self.g2[x2]
        else:
            output = self.g1(x1) + self.g2(x2)
        return output <= self.c
