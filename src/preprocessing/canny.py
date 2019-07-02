import cv2


def CannyThreshold(image, threshold=50, ratio=3, aperture_size=3):
    """
    Creates a contour map of the image using the Canny algorithm.
    :param image:
    :param threshold: first threshold for the hysteresis procedure
    :type: double
    :param ratio:
    :type: double
    :param aperture_size: aperture size for the Sobel operator
    :type: int
    :return: an image with the detected edges
    :type: np.ndarray
    """
    # Convert image to gray-scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Noise reduction: image is smoothed with a Gaussian filter
    detected_edges = cv2.GaussianBlur(gray, (5,5), 0)
    # Canny: image filtered with Sobel kernel both horizontally and vertically to find the gradient of the image in
    # both directions. Non-maximum suppression and histeresis thresholding are then applied.
    detected_edges = cv2.Canny(detected_edges, threshold, threshold * ratio, apertureSize=aperture_size)
    return detected_edges