from torch import no_grad, zeros
from torch.linalg import norm
import torch.nn.functional as F

class CPGD:
    def __init__(self, iterations, tolerance, num_classes=10):
        self.iterations = iterations
        self.tolerance = tolerance
        self.num_classes = num_classes
        self.mapping = self._get_mapping()

    def _get_mapping(self):
        """
        Prompt user for target class mappings.
        Returns a dictionary mapping source class -> target class
        """
        print("Please input Matrix Mapping Values")
        mapping = {}
        for i in range(self.num_classes):
            while True:
                try:
                    target = int(input(f"Class {i} -> "))
                    if 0 <= target < self.num_classes:
                        mapping[i] = target
                        break
                    else:
                        print(f"Error: Target must be between 0 and {self.num_classes-1}")
                except ValueError:
                    print("Error: Please enter a valid integer")
        return mapping

    def __call__(self, x, y, lr, model, loss):
        return self.cpgd(x, y, lr, model, loss)

    '''
    Controlled PGD implementation, executes a targeted attack based on mapping matrix

    @param x - the input images
    @param y - the true labels
    @param lr - the learning rate, hyper param of attack
    @param model - the model being attacked
    @param loss - callable loss, use loss of model being attacked
    @return the adversarial images
    '''
    def cpgd(self, x, y, lr, model, loss):
        step = x.clone().detach().requires_grad_(True)
        last_step = x.detach()
        
        # Create target labels based on mapping
        target_labels = self._get_target_labels(y)
        
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

    def _get_target_labels(self, y):
        """
        Convert true labels to target labels based on mapping matrix
        
        @param y - true labels (batch)
        @return target labels according to mapping
        """
        target_labels = zeros(y.size(), dtype=y.dtype, device=y.device)
        for i, label in enumerate(y):
            target_labels[i] = self.mapping[label.item()]
        return target_labels

    '''
    This is the projection step of the CPGD implementation
    
    @todo actually implement this, need to determine what a reasonable projection is
    '''
    def projection(self, a):
        return a.clone().detach().requires_grad_(True)