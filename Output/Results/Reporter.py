import torch
import os
from datetime import datetime

class BaseReporter:
    def report(self):
        raise NotImplementedError()

    def collect(self, results):
        raise NotImplementedError()
    
class SimpleAccReporter(BaseReporter):
    """
    Reporter for tracking attack success metrics.
    
    Tracks:
    - Global Attack Success Rate (GASR): Overall attack success
    - Individual Attack Success Rate (IASR): Per-class attack success
    - Accuracy: Classification accuracy (1 - GASR for untargeted, different for targeted)
    """
    
    def __init__(self, save_path=None, is_targeted=False, attack_params=None):
        self.total_misclassified = 0
        self.total_samples = 0
        self.class_misclassified = {}
        self.class_total = {}
        self.save_path = save_path
        self.is_targeted = is_targeted
        self.attack_params = attack_params or {}
    
    def collect(self, results):
        """
        Collect statistics from a batch.
        
        Args:
            results: AttackResults object with fields:
                    - num_misclassified: Number of misclassified samples
                    - batch_size: Size of the batch
                    - true_labels: True labels (optional, can be None)
                    - pred_labels: Predicted labels (optional, can be None)
        """
        # Extract values from AttackResults object
        num_misclassified = results.num_misclassified
        batch_size = results.batch_size
        true_labels = results.true_labels
        pred_labels = results.pred_labels
        
        # Update total statistics
        self.total_misclassified += num_misclassified.item() if torch.is_tensor(num_misclassified) else num_misclassified
        self.total_samples += batch_size
        
        # If labels are provided, track per-class statistics
        if true_labels is not None and pred_labels is not None:
            for true_label, pred_label in zip(true_labels, pred_labels):
                true_label = true_label.item() if torch.is_tensor(true_label) else true_label
                pred_label = pred_label.item() if torch.is_tensor(pred_label) else pred_label
                
                if true_label not in self.class_total:
                    self.class_total[true_label] = 0
                    self.class_misclassified[true_label] = 0
                
                self.class_total[true_label] += 1
                if true_label != pred_label:
                    self.class_misclassified[true_label] += 1
    
    def report(self):
        """
        Print and save the attack statistics to file.
        Only outputs for untargeted attacks (PGD).
        """
        # Skip reporting for targeted attacks
        if self.is_targeted:
            return
        
        # Prepare the report content
        report_lines = []
        report_lines.append("\n" + "="*60)
        report_lines.append("ATTACK RESULTS")
        report_lines.append("="*60)
    
        # Attack Parameters
        if self.attack_params:
            report_lines.append("\nAttack Parameters:")
            report_lines.append("-"*60)
            report_lines.append(f"Iterations: {self.attack_params.get('iterations', 'N/A')}")
            report_lines.append(f"Tolerance: {self.attack_params.get('tolerance', 'N/A')}")
            report_lines.append(f"Alpha (Step Size): {self.attack_params.get('alpha', 'N/A')}")
            report_lines.append(f"Epsilon (Max Perturbation): {self.attack_params.get('epsilon', 'N/A')}")
            report_lines.append("-"*60)
        
        # Global Attack Success Rate (GASR)
        gasr = (self.total_misclassified / self.total_samples) * 100 if self.total_samples > 0 else 0
        report_lines.append(f"Global Attack Success Rate (GASR): {gasr:.2f}%")
        
        # Accuracy (1 - GASR)
        accuracy = 100 - gasr
        report_lines.append(f"Accuracy: {accuracy:.2f}%")
        
        # Individual Attack Success Rate (IASR)
        if self.class_total:
            report_lines.append("\n" + "-"*60)
            report_lines.append("Individual Attack Success Rate (IASR) by Class:")
            report_lines.append("-"*60)
            
            for class_id in sorted(self.class_total.keys()):
                total = self.class_total[class_id]
                misclassified = self.class_misclassified[class_id]
                iasr = (misclassified / total) * 100 if total > 0 else 0
                report_lines.append(f"Class {class_id}: {iasr:.2f}% ({misclassified}/{total} misclassified)")
        
        report_lines.append("="*60 + "\n")
        
        # Print to console
        for line in report_lines:
            print(line)
        
        # Save to file if save_path is provided
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            with open(self.save_path, 'w') as f:
                f.write('\n'.join(report_lines))
            print(f"Results saved to: {self.save_path}")
    
    def reset(self):
        """Reset all statistics."""
        self.total_misclassified = 0
        self.total_samples = 0
        self.class_misclassified = {}
        self.class_total = {}


class TargetedSuccessReporter:
    """
    Reporter specifically for targeted attacks (CPGD).
    Tracks how well the attack achieved the targeted misclassifications.
    """
    
    def __init__(self, num_classes, mapping, save_path=None):
        self.num_classes = num_classes
        self.mapping = mapping
        self.save_path = save_path
        self.targeted_success = {}  # source_class -> {achieved: count, total: count}
        self.class_misclassified = {}  # Track misclassifications per class
        self.class_misclassified = {}
        
        # Initialize tracking for each source class
        for source in range(num_classes):
            self.targeted_success[source] = {
                'achieved_target': 0,
                'total': 0
            }
            self.class_misclassified[source] = 0
    
    def collect(self, true_labels, pred_labels, target_labels):
        """
        Collect targeted attack statistics.
        
        Args:
            true_labels: Original true labels
            pred_labels: Predicted labels after attack
            target_labels: Intended target labels from mapping
        """
        for true_label, pred_label, target_label in zip(true_labels, pred_labels, target_labels):
            true_label = true_label.item() if torch.is_tensor(true_label) else true_label
            pred_label = pred_label.item() if torch.is_tensor(pred_label) else pred_label
            target_label = target_label.item() if torch.is_tensor(target_label) else target_label
            
            self.targeted_success[true_label]['total'] += 1
            
            # Track misclassifications (pred != true)
            if pred_label != true_label:
                self.class_misclassified[true_label] += 1
            
            # Check if attack achieved the target
            if pred_label == target_label:
                self.targeted_success[true_label]['achieved_target'] += 1
    
    def report(self):
        """Print and save targeted attack statistics."""
        # Calculate overall metrics
        total_targeted_success = 0
        total_misclassified = 0
        total_samples = 0
        
        for source_class in self.targeted_success:
            stats = self.targeted_success[source_class]
            total_targeted_success += stats['achieved_target']
            total_misclassified += self.class_misclassified[source_class]
            total_samples += stats['total']
        
        overall_gasr = (total_targeted_success / total_samples) * 100 if total_samples > 0 else 0
        overall_accuracy = (1 - total_misclassified / total_samples) * 100 if total_samples > 0 else 0
        
        # Prepare the report content
        report_lines = []
        report_lines.append("\n" + "="*60)
        report_lines.append("TARGETED ATTACK RESULTS (CPGD)")
        report_lines.append("="*60)
    
        # Attack Parameters
        if self.attack_params:
            report_lines.append("\nAttack Parameters:")
            report_lines.append("-"*60)
            report_lines.append(f"Iterations: {self.attack_params.get('iterations', 'N/A')}")
            report_lines.append(f"Tolerance: {self.attack_params.get('tolerance', 'N/A')}")
            report_lines.append(f"Alpha (Step Size): {self.attack_params.get('alpha', 'N/A')}")
            report_lines.append(f"Epsilon (Max Perturbation): {self.attack_params.get('epsilon', 'N/A')}")
            report_lines.append("-"*60)
        
        # Global metrics
        report_lines.append(f"Global Attack Success Rate (GASR): {overall_gasr:.2f}%")
        report_lines.append(f"Accuracy: {overall_accuracy:.2f}%")
        
        # Individual Class Accuracy (misclassification stats)
        report_lines.append("\n" + "-"*60)
        report_lines.append("Individual Class Accuracy:")
        report_lines.append("-"*60)
        
        for source_class in sorted(self.targeted_success.keys()):
            stats = self.targeted_success[source_class]
            total = stats['total']
            misclassified = self.class_misclassified[source_class]
            misclassification_rate = (misclassified / total) * 100 if total > 0 else 0
            report_lines.append(f"Class {source_class}: {misclassification_rate:.2f}% ({misclassified}/{total} misclassified)")
        
        report_lines.append("-"*60)
        
        # Individual Attack Success Rate by class
        report_lines.append("\n" + "-"*60)
        report_lines.append("Individual Attack Success Rate (IASR) by Class:")
        report_lines.append("-"*60)
        report_lines.append(f"{'Class':<8} {'Target':<8} {'Success Rate':<15} {'Samples'}")
        report_lines.append("-"*60)
        
        for source_class in sorted(self.targeted_success.keys()):
            stats = self.targeted_success[source_class]
            total = stats['total']
            achieved = stats['achieved_target']
            
            if total > 0:
                success_rate = (achieved / total) * 100
                target_class = self.mapping[source_class]
                report_lines.append(f"{source_class:<8} {target_class:<8} {success_rate:>6.2f}%         {achieved}/{total} correctly targeted")
        
        report_lines.append("="*60 + "\n")
        
        # Print to console
        for line in report_lines:
            print(line)
        
        # Save to file if save_path is provided
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            with open(self.save_path, 'w') as f:
                f.write('\n'.join(report_lines))
            print(f"Results saved to: {self.save_path}")