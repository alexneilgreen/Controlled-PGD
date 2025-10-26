from torch import device, cuda

from Attack.PGD import PGD
from Result.Reporter import SimpleAccReporter

dev = device("cuda" if cuda.is_available() else "cpu")

class UntargetedAttack:
    def __init__(self, model, loss, dataloader, **kwargs):
        self.model = model
        self.loss = loss
        self.dataloader = dataloader

        iterations = 100    # some default? update this if needs be
        tolerance = .000001 # again, some default.  this probably will need to be different
        self.lr = .0000001
        for key in ('tolerance', 'iterations'):
            # bad implementation, not really scaleable, but it is only 3 options
            if key in kwargs:
                if key == 'tolerance':
                    tolerance = kwargs[key]
                elif key == 'iterations':
                    iterations = kwargs[key]
                elif key == 'lr':
                    self.lr = kwargs[key]

        self.pgd = PGD(iterations, tolerance)
        self.reporter = SimpleAccReporter()


    def execute_attack(self):
        for data, label in self.dataloader:
            data = data.to(device=dev)
            label = label.to(device=dev)
            advx = self.pgd(data, label, self.lr, self.model, self.loss)
            _, advlabel = self.model(advx).max(1)
            self.reporter.collect(((advlabel != label).sum(), advx.size(dim=0)))

        self.reporter.report()
