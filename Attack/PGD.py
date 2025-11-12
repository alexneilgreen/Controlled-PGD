from torch import no_grad
from torch.linalg import norm

class PGD:
    def __init__(self, iterations=100, tolerance=0.000001, epsilon=0.3, alpha=0.01):
        """
        Initialize PGD attack.
        
        Args:
            iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            epsilon: Maximum perturbation (L-infinity norm)
            alpha: Step size for each iteration
        """
        self.iterations = iterations
        self.tolerance = tolerance
        self.epsilon = epsilon
        self.alpha = alpha

    def __call__(self, x, y, lr, model, loss):
        return self.pgd(x, y, lr, model, loss)

    '''
    Base PGD implementation, executes an untargeted attack on input and returns

    @param x - the input images
    @param y - the true labels
    @param lr - the learning rate, hyper param of attack
    @param model - the model being attacked
    @param loss - callable loss, use loss of model being attacked
    @return the adversarial images
    '''
    def pgd(self, x, y, lr, model, loss):
        step = x.clone().detach().requires_grad_(True)
        last_step = x.detach()

        for _ in range(self.iterations):
            # calculate predicted labels
            pred = model(step)

            gradient = loss(pred, y)

            # calculate the output of the model
            model.zero_grad()
            gradient.backward()

            with no_grad():
                unproj_step = step - lr * gradient
                step = self.projection(unproj_step)
                
                if norm(step - last_step) < self.tolerance:
                    break
                last_step = step.detach()

        return step

    '''
    This is the projection step of the PGD implementation
    
    @todo actually implement this, need to determine what a reasonable projection is
    '''
    def projection(self, a):
        return a.clone().detach().requires_grad_(True)