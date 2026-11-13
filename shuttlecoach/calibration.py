"""Fit the court from visible, named painted-line intersections."""
import cv2
import numpy as np

from .upstream import court

# These intersections exist on painted lines; the net is not a floor landmark.
LANDMARKS = {}
for row, y in [("Near baseline", 0.), ("Near doubles long service", .76),
               ("Near short service", court.NET_Y_M - 1.98),
               ("Far short service", court.NET_Y_M + 1.98),
               ("Far doubles long service", court.COURT_LENGTH_M - .76),
               ("Far baseline", court.COURT_LENGTH_M)]:
    for column, x in [("left doubles sideline", 0.), ("left singles sideline", .46),
                      ("centre line", 3.05), ("right singles sideline", 5.64),
                      ("right doubles sideline", 6.10)]:
        LANDMARKS[f"{row} × {column}"] = (x, y)


def fitted_court_points(calibration):
    if calibration.get('mode', 'corners') == 'corners':
        return court.COURT_CORNERS_M
    if calibration.get('mode') != 'intersections':
        raise ValueError('Unknown calibration method.')
    try:
        return np.asarray([LANDMARKS[p['landmark']] for p in calibration.get('points', [])], dtype=float).reshape(-1, 2)
    except (KeyError, TypeError) as error:
        raise ValueError('Choose named painted-line intersections from the court diagram.') from error


def fit_calibration(calibration, meta):
    """Return image-to-metres H and outer corners, allowing inferred corners off-screen."""
    target = fitted_court_points(calibration)
    intersections = calibration.get('mode') == 'intersections'
    if intersections:
        try:
            pixels = np.asarray([[p['image_x'], p['image_y']] for p in calibration.get('points', [])], dtype=float).reshape(-1, 2)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('Each intersection needs finite source-image coordinates.') from error
        if len(pixels) < 4:
            raise ValueError('Mark at least four different visible intersections spanning court width and depth.')
        if len(np.unique(target, axis=0)) != len(target) or len(np.unique(pixels, axis=0)) != len(pixels):
            raise ValueError('Use each intersection and image point only once.')
    else:
        pixels = np.asarray(calibration.get('corners_px', []), dtype=float)
        if pixels.shape != (4, 2):
            raise ValueError('Enter four finite pixel coordinates: near-left, near-right, far-right, far-left.')
    if not np.isfinite(pixels).all():
        raise ValueError('Calibration points must be finite source-image coordinates.')
    if (pixels < 0).any() or (pixels[:, 0] >= meta['width']).any() or (pixels[:, 1] >= meta['height']).any():
        raise ValueError('Clicked calibration points must lie inside the source image.')
    if intersections:
        if np.ptp(target, axis=0).min() < 2 or cv2.contourArea(cv2.convexHull(target.astype(np.float32))) < 2:
            raise ValueError('Spread intersections across at least two metres of court width and depth; avoid a single line.')
        if cv2.contourArea(cv2.convexHull(pixels.astype(np.float32))) < meta['width'] * meta['height'] * .01:
            raise ValueError('Spread visible intersections over a larger part of the image, not along one line.')
        h, _ = cv2.findHomography(pixels, target, method=0)
        if h is None or not np.isfinite(h).all() or np.linalg.matrix_rank(h) < 3:
            raise ValueError('These intersections cannot define a stable court. Check their names and positions.')
        if np.linalg.norm(court.image_to_court(pixels, h) - target, axis=1).max() > .25:
            raise ValueError('The marked intersections disagree by more than 0.25 m. Check point names and clicks.')
        inverse = np.linalg.inv(h)
        denominator = np.c_[court.COURT_CORNERS_M, np.ones(4)] @ inverse[2]
        if not ((denominator > 0).all() or (denominator < 0).all()):
            raise ValueError('The fitted court crosses the camera horizon. Check the selected intersections.')
        corners = court.image_to_court(court.COURT_CORNERS_M, inverse)
        if not np.isfinite(corners).all() or np.abs(corners).max() > 20 * max(meta['width'], meta['height']):
            raise ValueError('The inferred court extends implausibly far outside the image. Check the intersections.')
    else:
        corners = pixels
        h = court.compute_homography(corners)
    contour = corners.astype(np.float32)
    if not cv2.isContourConvex(contour) or cv2.contourArea(contour) < meta['width'] * meta['height'] * .01:
        raise ValueError('Court corners must form a non-crossing, substantial quadrilateral.')
    if not intersections and not (corners[0, 0] < corners[1, 0] and corners[3, 0] < corners[2, 0]
                                  and corners[:2, 1].mean() > corners[2:, 1].mean()):
        raise ValueError('Use a rear-court view and click near-left, near-right, far-right, far-left in order.')
    return h, corners
