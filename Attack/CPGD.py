from torch import no_grad, zeros
from torch.linalg import norm
import torch

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

    def set_mapping(self, mapping):
        """Set the class mapping for targeted attacks."""
        self.mapping = mapping

    def __call__(self, x, y, lr, model, loss):
        return self.cpgd(x, y, lr, model, loss)

    def cpgd(self, x, y, lr, model, loss):
        """
        Controlled PGD implementation, executes a targeted attack based on mapping matrix

        @param x - the input images
        @param y - the true labels
        @param lr - the learning rate, hyper param of attack
        @param model - the model being attacked
        @param loss - callable loss, use loss of model being attacked
        @return the adversarial images
        """
        step = x.clone().detach().requires_grad_(True)
        last_step = x.detach()
        
        # Create target labels based on mapping
        target_labels = self.get_target_labels(y)
        
        for _ in range(self.iterations):
            # calculate predicted labels
            pred = model(step)
            
            # Use negative loss to maximize probability of target class
            # This makes the model think the image belongs to the target class
            gradient = -loss(pred, target_labels)
            
            # calculate the gradient
            model.zero_grad()
            gradient.backward()
            
            with no_grad():
                # Move in direction that increases target class probability
                unproj_step = step - lr * step.grad
                step = self.projection(unproj_step)
                
                if norm(step - last_step) < self.tolerance:
                    break
                last_step = step.detach()
                step = step.detach().requires_grad_(True)

        return step

    def get_target_labels(self, y):
        """
        Convert true labels to target labels based on mapping matrix
        
        @param y - true labels (batch)
        @return target labels according to mapping
        """
        target_labels = zeros(y.size(), dtype=y.dtype, device=y.device)
        for i, label in enumerate(y):
            target_labels[i] = self.mapping[label.item()]
        return target_labels

    def projection(self, a):
        """
        This is the projection step of the CPGD implementation
        
        @todo actually implement this, need to determine what a reasonable projection is
        """
        return a.clone().detach().requires_grad_(True)