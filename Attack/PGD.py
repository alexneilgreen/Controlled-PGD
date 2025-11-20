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

    def __call__(self, x, y, alpha, model, loss):
        return self.pgd(x, y, alpha, model, loss)

    def pgd(self, x, y, alpha, model, loss):
        '''
        Base PGD implementation, executes an untargeted attack on input and returns

        @param x - the input images
        @param y - the true labels
        @param alpha - attack step size, hyper param of attack
        @param model - the model being attacked
        @param loss - callable loss, use loss of model being attacked
        @return the adversarial images
        '''
        x_orig = x.clone().detach()  # Store original images
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
                unproj_step = step + alpha * step.grad  # Maximize loss (untargeted)
                step = self.projection(unproj_step, x_orig)
                
                if norm(step - last_step) < self.tolerance:
                    break
                last_step = step.detach()

        return step

    def projection(self, x_adv, x_orig):
        """
        Project adversarial example to be within epsilon ball of original image.
        Uses L-infinity norm constraint.
        
        @param x_adv - the adversarial images
        @param x_orig - the original clean images
        @return projected adversarial images
        """
        import torch
        # Compute perturbation
        perturbation = x_adv - x_orig
        
        # Clip perturbation to [-epsilon, epsilon]
        perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
        
        # Add perturbation back to original and clip to valid image range [0, 1]
        x_projected = x_orig + perturbation
        x_projected = torch.clamp(x_projected, 0, 1)
        
        return x_projected.clone().detach().requires_grad_(True)