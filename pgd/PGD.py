
class PGD:
    def __init__(self, iterations, tolerance):
        self.iterations = iterations
        self.tolerance = tolerance

    '''
    Base PGD implementation, executes an untargeted attack on input and returns

    \param x - the input images
    \param lr - the learning rate, hyper param of attack
    \param loss - callable loss, use loss of model being attacked
    \return the adversarial images
    '''
    def pgd(self, x, lr, loss):
        step = x
        last_step = x.copy()
        for i in range(self.iterations):
            gradient = loss(step)
            unproj_step = step - lr * gradient
            step = self.projection(unproj_step)
            if step - last_step < self.tolerance:
                break
            last_step = step

        return step

    '''
    This is the projection step of the PGD implementation
    
    \todo actually implement this, need to determine what a reasonable projection is
    '''
    def projection(self, a):
        return a