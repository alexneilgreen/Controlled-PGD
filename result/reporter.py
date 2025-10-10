class BaseReporter:
    def report(self):
        raise NotImplementedError()

    def collect(self, results):
        raise NotImplementedError()
    
class SimpleAccReporter(BaseReporter):
    def __init__(self):
        self.success = 0
        self.total = 0

    def report(self):
        # exception as logical control.  dislike, but fine for now
        if self.success > self.total:
            raise ValueError("Succeeded more times than tried.  What?")
        elif self.total == 0:
            raise ZeroDivisionError("No attmepts! Cannot report!")
        
        acc = self.success / self.total
        print(f"Accuracy of model: {acc}")

    def collect(self, results):
        if not isinstance(results, tuple):
            raise ValueError("results should be of form (successes, totals)")
        success, total = results
        self.success += success
        self.total += total
