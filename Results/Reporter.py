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
    - Accuracy: Classification accuracy (1 - GASR)
    """
    
    def __init__(self, save_path=None):
        self.total_misclassified = 0
        self.total_samples = 0
        self.class_misclassified = {}
        self.class_total = {}
        self.save_path = save_path
    
    def collect(self, results):
        """
        Collect statistics from a batch.
        
        Args:
            results: Tuple of (num_misclassified, batch_size) or
                     Tuple of (num_misclassified, batch_size, true_labels, pred_labels)
        """
        if len(results) == 2:
            num_misclassified, batch_size = results
            self.total_misclassified += num_misclassified.item() if torch.is_tensor(num_misclassified) else num_misclassified
            self.total_samples += batch_size
        elif len(results) == 4:
            num_misclassified, batch_size, true_labels, pred_labels = results
            self.total_misclassified += num_misclassified.item() if torch.is_tensor(num_misclassified) else num_misclassified
            self.total_samples += batch_size
            
            # Track per-class statistics
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
        """
        # Prepare the report content
        report_lines = []
        report_lines.append("\n" + "="*60)
        report_lines.append("ATTACK RESULTS")
        report_lines.append("="*60)
        
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
        self.targeted_success = {}  # source_class -> {target: count, total: count}
        
        # Initialize tracking for each source class
        for source in range(num_classes):
            self.targeted_success[source] = {
                'achieved_target': 0,
                'total': 0
            }
    
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
            
            # Check if attack achieved the target
            if pred_label == target_label:
                self.targeted_success[true_label]['achieved_target'] += 1
    
    def report(self):
        """Print and save targeted attack statistics."""
        # Prepare the report content
        report_lines = []
        report_lines.append("\n" + "="*60)
        report_lines.append("TARGETED ATTACK SPECIFICS (CPGD)")
        report_lines.append("="*60)
        
        total_targeted_success = 0
        total_samples = 0
        
        report_lines.append("\nTargeted Success Rate by Class:")
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
                report_lines.append(f"{source_class:<8} {target_class:<8} {success_rate:>6.2f}%         {achieved}/{total}")
                
                total_targeted_success += achieved
                total_samples += total
        
        report_lines.append("-"*60)
        
        if total_samples > 0:
            overall_targeted_success = (total_targeted_success / total_samples) * 100
            report_lines.append(f"\nOverall Targeted Success Rate: {overall_targeted_success:.2f}%")
            report_lines.append(f"(Percentage of samples that were misclassified to the intended target)")
        
        report_lines.append("="*60 + "\n")
        
        # Print to console
        for line in report_lines:
            print(line)
        
        # Save to file if save_path is provided
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            # Append to existing file
            with open(self.save_path, 'a') as f:
                f.write('\n'.join(report_lines))
            print(f"Targeted results appended to: {self.save_path}")