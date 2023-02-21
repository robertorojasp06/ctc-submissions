"""
Module with classes to evaluate detection performance.
"""
import numpy as np
from tqdm import tqdm

from config.constants import VOXEL_SIZE

# distance-to-target tolerance (in microns)
TOLERANCE = 2.04


def compute_distance(target, detection):
    """Return the distance in microns between 'target' and 'detection'.

    Parameters
    ----------
    target : array
        Coordinates of the target expressed as [slice, row, column].
    detection : array
        Coordinates of the detection expressed as [slice, row, column].
    """
    return np.sqrt(
        VOXEL_SIZE["slice"] ** 2 * (target[0] - detection[0]) ** 2 +
        VOXEL_SIZE["row"] ** 2 * (target[1] - detection[1]) ** 2 +
        VOXEL_SIZE["column"] ** 2 * (target[2] - detection[2]) ** 2
    )


def get_detection_object(match):
    """Helper function to map a list of matches to a list of its
    detection objects"""
    return match.detection


class Detection:
    """Class for detections."""
    def __init__(self, coordinates):
        self.coordinates = coordinates

    def __eq__(self, other):
        return all(self.coordinates == other.coordinates)


class Target:
    """Class for detection target."""
    def __init__(self, coordinates, label):
        self.coordinates = coordinates
        self.label = label
        self.enclosed_detections = None

    def __eq__(self, other):
        return all(self.coordinates == other.coordinates) and (self.label == other.label)

    @property
    def nearest_detection(self):
        if self.enclosed_detections:
            return sorted(
                self.enclosed_detections,
                key=lambda x: compute_distance(self.coordinates, x.coordinates))[0]


class Match:
    """Class for matches between targets and detections."""
    def __init__(self, target, detection):
        self.target = target
        self.detection = detection

    @property
    def distance(self):
        return compute_distance(
            self.target.coordinates, self.detection.coordinates)


class VolumeEvaluator:
    """Class to evaluate detection performance over a volume.
    
    Parameters
    ----------
    targets : dict
        Volume targets information extracted from csv file.
    detections : list
        Volume detections expressed as arrays of coordinates
        [slice, row, column].
    tolerance_radius : float, optional
        Distance in microns of the searching ball.
        Detections with a distance-to-target smaller than
        'tolerance_radius' (in microns) are accepted as target
        detections.
    """
    def __init__(self, targets, detections, tolerance_radius=TOLERANCE):
        self.targets = targets
        self.detections = detections
        self.tolerance_radius = tolerance_radius
        self._matches = None
        self._performance = None

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, targets):
        self._targets = []
        for slice, row, col, label in zip(targets["centroid (slice)"],
                                          targets["centroid (row)"],
                                          targets["centroid (column)"],
                                          targets["label"]):
            self._targets.append(Target(np.array([slice, row, col]), label))

    @property
    def detections(self):
        return self._detections

    @detections.setter
    def detections(self, detections):
        self._detections = []
        for coordinates in detections:
            self._detections.append(Detection(coordinates))

    @property
    def matches(self):
        return self._matches

    @property
    def performance(self):
        if self._performance:
            distances = [match.distance for match in self.matches]
            return {
                "detections_count": len(self.matches),
                "detections_ratio": len(self.matches) * 1.0 / len(self.targets),
                "distances": distances
            }

    def run_evaluation(self, verbose=False):
        """Find matches between targets and detections considering the
        tolerance radius and using a minimum-distance criteria to break
        possible matches with the same detection."""
        candidates = []
        self._matches = []
        if verbose:
            print("Finding candidates ...")
        for target in tqdm(self.targets, disable=not verbose):
            enclosed = [
                detection
                for detection in self.detections
                if compute_distance(
                    target.coordinates,
                    detection.coordinates) <= self.tolerance_radius
            ]
            if enclosed:
                target.enclosed_detections = enclosed
                candidates.append(Match(target, target.nearest_detection))
        if verbose:
            print("Updating candidates by min-distance criteria ...")
        candidates = sorted(candidates, key=lambda x: x.distance)
        for candidate in tqdm(candidates, disable=not verbose):
            for detection in sorted(
                candidate.target.enclosed_detections,
                key=lambda x: compute_distance(candidate.target.coordinates, x.coordinates)
            ):
                if detection not in map(get_detection_object, self._matches):
                    self._matches.append(Match(candidate.target, detection))
                    break
        self._performance = True