import cv2
import numpy as np
import os


def findHomography(image_1_kp, image_2_kp, matches):
    """Returns the homography describing the transformation between the
    keypoints of image 1 and image 2.

        ************************************************************
          Before you start this function, read the documentation
                  for cv2.DMatch, and cv2.findHomography
        ************************************************************

    Follow these steps:

        1. Iterate through matches and store the coordinates for each
           matching keypoint in the corresponding array (e.g., the
           location of keypoints from image_1_kp should be stored in
           image_1_points).

            NOTE: Image 1 is your "query" image, and image 2 is your
                  "train" image. Therefore, you index into image_1_kp
                  using `match.queryIdx`, and index into image_2_kp
                  using `match.trainIdx`.

        2. Call cv2.findHomography() and pass in image_1_points and
           image_2_points, using method=cv2.RANSAC and
           ransacReprojThreshold=5.0.

        3. cv2.findHomography() returns two values: the homography and
           a mask. Ignore the mask and return the homography.

    Parameters
    ----------
    image_1_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors in the first image

    image_2_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors in the second image

    matches : list<cv2.DMatch>
        A list of matches between the keypoint descriptor lists

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        A 3x3 array defining a homography transform between image_1 and image_2
    """
    # image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    # image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    image_1_points = np.array(map(lambda match: image_1_kp[match.queryIdx].pt, matches))
    image_2_points = np.array(map(lambda match: image_2_kp[match.trainIdx].pt, matches))
    h, _ = cv2.findHomography(image_1_points, image_2_points, cv2.RANSAC, 5.0)
    return h


def drawMatches(image_1, image_1_keypoints, image_2, image_2_keypoints, matches):
    """ Draws the matches between the image_1 and image_2.

    Note: Do not edit this function, it is provided for you for visualization
    purposes.

    Args:
    image_1 (numpy.ndarray): The first image (can be color or grayscale).
    image_1_keypoints (list): The image_1 keypoints, the elements are of type
                              cv2.KeyPoint.
    image_2 (numpy.ndarray): The image to search in (can be color or grayscale)
    image_2_keypoints (list): The image_2 keypoints, the elements are of type
                              cv2.KeyPoint.

    Returns:
    output (numpy.ndarray): An output image that draws lines from the input
                            image to the output image based on where the
                            matching features are.
    """
    # Compute number of channels.
    num_channels = 1
    if len(image_1.shape) == 3:
        num_channels = image_1.shape[2]
    # Separation between images.
    margin = 10
    # Create an array that will fit both images (with a margin of 10 to
    # separate the two images)
    joined_image = np.zeros((max(image_1.shape[0], image_2.shape[0]),
                             image_1.shape[1] + image_2.shape[1] + margin,
                             3))
    if num_channels == 1:
        for channel_idx in range(3):
            joined_image[:image_1.shape[0],
            :image_1.shape[1],
            channel_idx] = image_1
            joined_image[:image_2.shape[0],
            image_1.shape[1] + margin:,
            channel_idx] = image_2
    else:
        joined_image[:image_1.shape[0], :image_1.shape[1]] = image_1
        joined_image[:image_2.shape[0], image_1.shape[1] + margin:] = image_2

    for match in matches:
        image_1_point = (int(image_1_keypoints[match.queryIdx].pt[0]),
                         int(image_1_keypoints[match.queryIdx].pt[1]))
        image_2_point = (int(image_2_keypoints[match.trainIdx].pt[0] +
                             image_1.shape[1] + margin),
                         int(image_2_keypoints[match.trainIdx].pt[1]))

        rgb = (np.random.rand(3) * 255).astype(np.int)
        cv2.circle(joined_image, image_1_point, 5, rgb, thickness=-1)
        cv2.circle(joined_image, image_2_point, 5, rgb, thickness=-1)
        cv2.line(joined_image, image_1_point, image_2_point, rgb, thickness=3)

    return joined_image


def findMatchesBetweenImages(image_1, image_2, num_matches):
    """Return the top list of matches between two input images.

    Parameters
    ----------
    image_1 : numpy.ndarray
        The first image (can be a grayscale or color image)

    image_2 : numpy.ndarray
        The second image (can be a grayscale or color image)

    num_matches : int
        The number of keypoint matches to find. If there are not enough,
        return as many matches as you can.

    Returns
    -------
    image_1_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors from image_1

    image_2_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors from image_2

    matches : list<cv2.DMatch>
        A list of the top num_matches matches between the keypoint descriptor
        lists from image_1 and image_2

    Notes
    -----
        (1) You will not be graded for this function. This function is almost
        identical to the function in Assignment 7 (we just parametrized the
        number of matches). We expect you to use the function you wrote in
        A7 here.

        (2) Python functions can return multiple values by listing them
        separated by spaces. Ex.

            def foo():
                return [], [], []
    """
    orb = cv2.ORB(nfeatures=500, scaleFactor=1.5, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE,
                  patchSize=31)

    image_1_kp, des1 = orb.detectAndCompute(image_1, None)
    image_2_kp, des2 = orb.detectAndCompute(image_2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # We coded the return statement for you. You are free to modify it -- just
    # make sure the tests pass.
    return image_1_kp, image_2_kp, matches[0:num_matches]


def weightAndSharpen(images):
    a = average(*images)
    intense = 2 * a
    blurred = cv2.GaussianBlur(a, ksize=(51, 51), sigmaY=9, sigmaX=9)
    return intense - blurred


def average(*images):
    # Need to prevent int overflow
    return reduce(lambda x, y: (x / 1.) + (y / 1.), images) / len(images)


def align(image1, image2, num_matches = 30):
    rows, cols, _ = image2.shape
    img1_kp, img2_kp, matches = findMatchesBetweenImages(image1, image2, num_matches)

    homo = findHomography(img1_kp, img2_kp, matches)
    return (cv2.warpPerspective(image1, homo, (cols, rows))), img1_kp, img2_kp, matches


def alignImages(images, num_matches=30):
    image2 = images[1]
    return map(lambda image: align(image, image2, num_matches), images)


def gBlur(mag, ksize=51, sigma=9):
    return cv2.GaussianBlur(mag, ksize=(ksize, ksize), sigmaY=sigma, sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)


def boxBlur(mag, k_size=51):
    smoother = np.ones((k_size, k_size)) / (k_size ** 2)
    return cv2.filter2D(mag, cv2.CV_64F, smoother)


def allInFocus(images, blurFunc=gBlur):
    ### Calculate gradient magnitude
    mags = magnitudes(images)
    ### Blur magnitude
    blurs = map(blurFunc, mags)
    ### Create a mask for each image and create new image
    maskMaker = np.array(blurs).argmax(axis=0)
    new_image = np.zeros(images[0].shape)
    masks = []
    for image_order in range(len(images)):
        index = (maskMaker == image_order)
        binaryImage = index.astype("uint8") * 255
        masks.append(binaryImage)
        new_image[index] = images[image_order][index]

    return new_image, mags, blurs, masks


def writeWithIndex(images, base_name, output_dir):
    i = 0
    for image in images:
        cv2.imwrite(os.path.join(output_dir, base_name + str(i) + ".jpg"), image)
        i += 1

def magnitudes(images):
    return map(magnitude, images)


def magnitude(image):
    if len(image.shape) == 3:
        imagez = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        imagez = image
    sobelx = cv2.Sobel(imagez, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(imagez, cv2.CV_64F, 0, 1)
    return cv2.magnitude(sobelx, sobely)
