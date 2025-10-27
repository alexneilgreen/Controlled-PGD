from torch import device, cuda
import torch

from Attack.PGD import PGD
from Attack.CPGD import CPGD
from Results.Reporter import SimpleAccReporter, TargetedSuccessReporter

dev = device("cuda" if cuda.is_available() else "cpu")

class UntargetedAttack:
    def __init__(self, model, loss, dataloader, save_path=None, **kwargs):
        self.model = model
        self.loss = loss
        self.dataloader = dataloader
        self.save_path = save_path

        iterations = kwargs.get('iterations', 100)
        tolerance = kwargs.get('tolerance', 0.000001)
        epsilon = kwargs.get('epsilon', 0.3)
        alpha = kwargs.get('alpha', 0.01)
        
        # Override with lr if provided (for backward compatibility)
        if 'lr' in kwargs:
            alpha = kwargs['lr']

        self.pgd = PGD(iterations=iterations, tolerance=tolerance, 
                      epsilon=epsilon, alpha=alpha)
        self.reporter = SimpleAccReporter(save_path=save_path)

    def execute_attack(self):
        print("\nExecuting PGD Attack...")
        print(f"Processing {len(self.dataloader)} batches...")
        
        for batch_idx, (data, label) in enumerate(self.dataloader):
            data = data.to(device=dev)
            label = label.to(device=dev)
            
            # Generate adversarial examples
            advx = self.pgd(data, label, self.pgd.alpha, self.model, self.loss)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(advx)
                _, advlabel = outputs.max(1)
            
            # Collect statistics with class information
            misclassified = (advlabel != label)
            self.reporter.collect((
                misclassified.sum(),
                advx.size(dim=0),
                label.cpu(),
                advlabel.cpu()
            ))
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(self.dataloader)} batches")

        self.reporter.report()


class TargetedAttack:
    def __init__(self, model, loss, dataloader, num_classes=10, mapping=None, save_path=None, **kwargs):
        self.model = model
        self.loss = loss
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.save_path = save_path

        iterations = kwargs.get('iterations', 100)
        tolerance = kwargs.get('tolerance', 0.000001)
        epsilon = kwargs.get('epsilon', 0.3)
        alpha = kwargs.get('alpha', 0.01)
        
        # Override with lr if provided (for backward compatibility)
        if 'lr' in kwargs:
            alpha = kwargs['lr']

        self.cpgd = CPGD(iterations=iterations, tolerance=tolerance, 
                        num_classes=num_classes, epsilon=epsilon, alpha=alpha)
        
        # Set mapping if provided
        if mapping is not None:
            self.cpgd.set_mapping(mapping)
        
        self.reporter = SimpleAccReporter(save_path=save_path)
        self.targeted_reporter = TargetedSuccessReporter(num_classes, mapping if mapping else self.cpgd.mapping, save_path=save_path)

    def execute_attack(self):
        print("\nExecuting CPGD Attack...")
        print(f"Processing {len(self.dataloader)} batches...")
        
        for batch_idx, (data, label) in enumerate(self.dataloader):
            data = data.to(device=dev)
            label = label.to(device=dev)
            
            # Generate adversarial examples
            advx = self.cpgd(data, label, self.cpgd.alpha, self.model, self.loss)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(advx)
                _, advlabel = outputs.max(1)
            
            # Get target labels for this batch
            target_labels = self.cpgd.get_target_labels(label)
            
            # Collect general statistics
            misclassified = (advlabel != label)
            self.reporter.collect((
                misclassified.sum(),
                advx.size(dim=0),
                label.cpu(),
                advlabel.cpu()
            ))
            
            # Collect targeted attack statistics
            self.targeted_reporter.collect(
                label.cpu(),
                advlabel.cpu(),
                target_labels.cpu()
            )
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(self.dataloader)} batches")

        self.reporter.report()
        self.targeted_reporter.report()