"""
Module with tools to measure segmentation performance
between output and reference masks.
"""
import sys
import numpy as np


class BinaryMetrics:
    """Metrics for measure semantic segmentation performance
    in a binary problem.
    Metrics 'dice' and 'jaccard' return numpy.nan if the reference
    image is full of negative values.

    Parameters
    ----------
    boolean_input : bool, optional
        Set to True to indicate that output and reference images
        are boolean. Set to False if images contain numeric values.
    positive_label : int, optional
        Positive label for images containing numeric values.
    negative_label : int, optional
        Negative label for images containing numeric values.
    """
    def __init__(self, boolean_input=True, positive_label=255,
                 negative_label=0):
        self.boolean_input = boolean_input
        self.positive_label = positive_label
        self.negative_label = negative_label

    def _check_values(self, image):
        uniques = sorted(np.unique(image))
        if (len(uniques) > 2 or
           (len(uniques) == 2 and
           (self.negative_label != uniques[0] or
            self.positive_label != uniques[1])) or
           (len(uniques) == 1 and
            uniques[0] not in [self.positive_label,
                               self.negative_label])):
            sys.exit(
                "Images are supposed to have only values"
                f" {self.negative_label} and {self.positive_label}"
            )

    def base_metrics(self, output, reference):
        metrics = {}
        if not self.boolean_input:
            for image in (output, reference):
                image = image.astype('int')
                self._check_values(image)
            output = output == self.positive_label
            reference = reference == self.positive_label
        metrics.update({
            'true_positives': np.logical_and(output, reference).sum()
        })
        metrics.update({
            'true_negatives': np.logical_and(~output, ~reference).sum()
        })
        metrics.update({
            'false_positives': output.sum() - metrics['true_positives']
        })
        metrics.update({
            'false_negatives': (~output).sum() - metrics['true_negatives']
        })
        return metrics

    def pixel_accuracy(self, output, reference, base_metrics=None):
        if not base_metrics:
            base_metrics = self.base_metrics(output, reference)
        correct = base_metrics['true_positives'] + base_metrics['true_negatives']
        all = (base_metrics['true_positives']
               + base_metrics['true_negatives']
               + base_metrics['false_positives'] 
               + base_metrics['false_negatives'])
        return correct / all

    def dice(self, output, reference, base_metrics=None):
        if not base_metrics:
            base_metrics = self.base_metrics(output, reference)
        num = 2 * base_metrics['true_positives']
        den = (2 * base_metrics['true_positives']
               + base_metrics['false_positives']
               + base_metrics['false_negatives'])
        # Special case: reference image is full of negatives
        if reference.sum() == 0:
            dice = np.nan
        else:
            dice = num / den
        return dice

    def jaccard(self, output, reference, base_metrics=None):
        if not base_metrics:
            base_metrics = self.base_metrics(output, reference)
        num = base_metrics['true_positives']
        den = (base_metrics['true_positives']
               + base_metrics['false_positives']
               + base_metrics['false_negatives'])
        # Special case: reference image is full of negatives
        if reference.sum() == 0:
            jaccard = np.nan
        else:
            jaccard = num / den
        return jaccard

    def all(self, output, reference):
        base_metrics = self.base_metrics(output, reference)
        base_metrics.update({
            'pixel_accuracy': self.pixel_accuracy(
                output,
                reference,
                base_metrics
            )
        })
        base_metrics.update({
            'dice': self.dice(
                output,
                reference,
                base_metrics
            )
        })
        base_metrics.update({
            'jaccard': self.jaccard(
                output,
                reference,
                base_metrics
            )
        })
        return base_metrics