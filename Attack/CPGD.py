import torch
from torch import no_grad, zeros
from torch.linalg import norm

class CPGD:
    def __init__(self, iterations=100, tolerance=0.000001, epsilon=0.3, alpha=0.01, num_classes=10, mapping=None):
        """
        Initialize CPGD (Controlled PGD) attack.
        
        Args:
            iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            epsilon: Maximum perturbation (L-infinity norm)
            alpha: Step size for each iteration
            num_classes: Number of classes in the dataset
            mapping: Dictionary mapping source class -> target class
        """
        self.iterations = iterations
        self.tolerance = tolerance
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_classes = num_classes
        self.mapping = mapping

    def __call__(self, x, y, alpha, model, loss):
        return self.cpgd(x, y, alpha, model, loss)

    def cpgd(self, x, y, alpha, model, loss):
        """
        Controlled PGD implementation, executes a targeted attack based on mapping matrix

        @param x - the input images
        @param y - the true labels
        @param alpha - attack step size, hyper param of attack
        @param model - the model being attacked
        @param loss - callable loss, use loss of model being attacked
        @return the adversarial images
        """
        x_orig = x.clone().detach()  # Store original images
        step = x.clone().detach().requires_grad_(True)
        last_step = step.detach().clone()
        
        # Create target labels based on mapping
        target_labels = self.get_target_labels(y)
        
        for _ in range(self.iterations):
            # calculate predicted labels
            pred = model(step)
            
            # Use negative loss to maximize probability of target class
            # This makes the model think the image belongs to the target class
            gradient = loss(pred, target_labels)
            
            # clear grads
            model.zero_grad()
            if step.grad is not None:
                step.grad.zero_()

            gradient.backward()
            grad = step.grad
            
            with no_grad():
                # Move in direction that increases target class probability
                unproj_step = step - alpha * grad.sign()
                step = self.projection(unproj_step, x_orig)
                
                # convergence check
                if (step - last_step).abs().max() < self.tolerance:
                    break

                last_step = step.detach().clone()

        return step.detach()

    def get_target_labels(self, y):
        """
        Vectorized mapping
        """
        return torch.tensor(
            [self.mapping[int(lbl)] for lbl in y],
            device=y.device,
            dtype=y.dtype
        )

    def projection(self, x_adv, x_orig):
        """
        Project adversarial example to be within epsilon ball of original image.
        Uses L-infinity norm constraint.
        
        @param x_adv - the adversarial images
        @param x_orig - the original clean images
        @return projected adversarial images
        """
        # Compute perturbation
        perturbation = x_adv - x_orig
        
        # Clip perturbation to [-epsilon, epsilon]
        perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
        
        # Add perturbation back to original and clip to valid image range [1, 1]
        x_projected = x_orig + perturbation
        x_projected = torch.clamp(x_projected, -1, 1)
        
        return x_projected.clone().detach().requires_grad_(True)