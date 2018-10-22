import os
import csv
import cv2
import copy
import random
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

STEERING_CORRECTION_LEFT  = 0.2
STEERING_CORRECTION_RIGHT = 0.2
USE_SIDE_CAMERAS    = True
FLIP_IMAGES         = True

class SteeringData:
    def __init__(self, path, angle, flipped):
        self.path    = path
        self.angle   = angle
        self.flipped = flipped
        self.shadow  = 0
        self.bright  = 0
        self.blur    = 0


def read_csv(csv_file):
    images = []
    angles = []
    print("Reading data from csv file....")
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)

        csv_data = []

        for line in reader:
            center_image_path = 'data/IMG/' + line[0].split('/')[-1]
            if os.path.isfile(center_image_path):
                center_angle = float(line[3])
                csv_data.append(SteeringData(center_image_path, center_angle, 0))
                if (FLIP_IMAGES):
                    csv_data.append(SteeringData(center_image_path, -center_angle, 1))

            if (USE_SIDE_CAMERAS):
                left_image_path   = 'data/IMG/' + line[1].split('/')[-1]
                if os.path.isfile(left_image_path):
                    left_angle   = float(line[3]) + STEERING_CORRECTION_LEFT
                    csv_data.append(SteeringData(left_image_path, left_angle, 0))
                    if (FLIP_IMAGES):
                        csv_data.append(SteeringData(left_image_path, -left_angle, 1))

                right_image_path   = 'data/IMG/' + line[2].split('/')[-1]
                if os.path.isfile(right_image_path):
                    right_angle   = float(line[3]) - STEERING_CORRECTION_RIGHT
                    csv_data.append(SteeringData(right_image_path, right_angle, 0))
                    if (FLIP_IMAGES):
                        csv_data.append(SteeringData(right_image_path, -right_angle, 1))

    print("Done. ")
    return shuffle(csv_data)


def get_bin_counts(x):
    #
    # First we must find the distribution of
    # steering angle corrections.
    # From this distribution we calculate the
    # number of images to augment.
    #
    corrections = [item.angle for item in x]

    #
    # We sort the steering angles into 25 bins
    # from the minimum steering angle to the max
    #
    bin_count = 25
    min_bin = np.min(corrections)
    max_bin = np.max(corrections)
    spread = max_bin - min_bin
    bin_size = spread / bin_count

    #
    # crete our bins
    #
    bins = [min_bin + i * bin_size for i in range(bin_count)]
    bins.append(max_bin + 0.10)

    #
    # np.histogram gives us the count of indices in
    # each bin.
    #
    bin_counts, bins_out = np.histogram(corrections, bins)

    #
    # The number we would like to see in each bin is the mean.
    #
    desired_count_per_bin = int(np.mean(bin_counts)) * 2


    return bins, bin_counts, desired_count_per_bin



#
# This method takes the dataset supplied by x
# and adds images.
# Existing images in the dataset are copied
# and augmented by a random blur, random
# shadows, and/or random brightness changes.
#
def augment_dataset(x, fix_dist):

    print("Augmenting dataset...")

    bins, bin_counts, desired_count_per_bin = get_bin_counts(x)

    #
    # Calculate the number of augmented images to generate from each file
    # in each of the bins. for example: each file in a bin containing 20 images
    # will each result in 5 augmentations when the desired_count_per_bin is 100.
    #
    augments = np.float32((desired_count_per_bin - bin_counts) / bin_counts)

    #
    # keep a running tally
    #
    augment_tally = np.zeros_like(augments)
    augmented = []
    for i in range(len(x)):
        steering_data = x[i]

        #
        # find the bin index and the number to augment.
        #
        augment_index = np.digitize(steering_data.angle, bins) - 1
        augment_tally[augment_index] += augments[augment_index]
        number_to_augment = np.int32(augment_tally[augment_index])

        #
        # Generate the data to augment.  Actual image modification
        # is done by the data generator when fitting the model.
        #
        for j in range(number_to_augment):
            #
            # copy the image data and randomly determine
            # the augmentations of shadow, blur, and brightness
            #
            new_steering_data =  copy.deepcopy(steering_data) #SteeringData(steering_data.path, steering_data.angle, steering_data.flipped)
            new_steering_data.shadow = int(np.random.uniform(0, 1) + 0.5)
            new_steering_data.blur   = int(np.random.uniform(0, 1) + 0.5)
            new_steering_data.bright = int(np.random.uniform(0, 1) + 0.5)
            augmented.append(new_steering_data)

        #
        # We have 'number_to_augment' less todo next time around.
        #
        augment_tally[augment_index] -= number_to_augment

    #
    # We fix the distribution and return the training set
    #
    if (fix_dist):
        return fix_distribution(x + augmented, bins, desired_count_per_bin, bin_counts)
    else:
        return x + augmented



#
# Since the track is mostly straignt with only slight curves the
# dataset is heavily skewed to those steering angles near zero. This
# method evens out that distribution
#
#
def fix_distribution(training_set, bins, bin_counts, desired_count_per_bin):

    print("Fixing Dataset Distribution...")

    #
    # Calculate the probability to keep any given item in a bin
    # based on the count above/below the desired count per bin.
    #
    #
    # ensure we don't divide by zero
    #
    non_zero_bincounts = np.array(bin_counts)
    non_zero_bincounts[non_zero_bincounts==0] = desired_count_per_bin
    keep_probabilities = (1 / (non_zero_bincounts / desired_count_per_bin) )

    def should_keep(item):
        #
        # calc the probability to keep this item based on
        # what bin it is in.
        #
        probability_to_keep = keep_probabilities[np.digitize(item.angle, bins) - 1]

        #
        # get a random percentage
        #
        random_prob = np.random.uniform(0, 1)

        #
        # if the random percentage is less that the probability
        # to keep we keep it. (return True)
        #
        return (random_prob <= probability_to_keep)

    #
    # Trim extra examples from the data set to fix distribution
    #
    trimmed_training_set = [x for x in training_set if should_keep(x)]
    return trimmed_training_set



#
# This method returns the training and validation dsat sets.
# Data is read from the CSV file at the given path.
# The training and vallidation sets are split
#
def load_data_sets(path, split = 0.2):
    steering_data = read_csv(path)
    train, valid = train_test_split(steering_data, test_size=split)
    return train, valid



def preprocess_image(img):
    #
    # img coming in is and RGB image of
    # shape: (160, 320, 3).  Convert to BGR,
    # Crop 50 pixels off the top of the image,
    # and 20 pixels off the bottom. Then
    # resize to 128, 128, 3
    #
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cropped = bgr[50:140, : ]
    resized = cv2.resize(cropped, (128, 128))
    return resized


def random_blur(image):
    #
    # Generate a random odd number for our
    # kernel size between 3 and 9
    #
    kernel_size = (np.random.randint(1, 5) * 2) + 1

    #
    # Blur and return
    #
    return cv2.GaussianBlur(image, (kernel_size, kernel_size),  0)


def random_shadow(image):
    height, width = image.shape[:2]

    #
    # First randomly determine a number of shadows
    # to apply to the image
    #
    number_of_shadows = np.random.randint(1, 6)

    #
    # Make our shadows within the image
    #
    list_of_shadows = []
    for index in range(number_of_shadows):
        #
        # The vertices in this shadow
        #
        shadow_vertices = []

        #
        # Shadows are at least a triangle but
        # could be a polygon with up to 25 points
        #
        number_of_points = np.random.randint(3, 26)

        #
        # Randomly determine the polygon points
        #
        for _ in range(number_of_points):
            random_point_x = width * np.random.uniform()
            random_point_y = height * np.random.uniform()
            shadow_vertices.append(( random_point_x, random_point_y))

        #
        # add to the list of all shadows applied to the image
        #
        list_of_shadows.append( np.array([shadow_vertices], dtype=np.int32))

        #
        # Create a mask with the same dimensions
        # as the original image
        #
        mask = np.zeros((height, width))

        #
        # On the mask, use opencv fillPoly()
        # to fill all the shadow polygon
        # area with white
        #
        for shadow in list_of_shadows:
            cv2.fillPoly(mask, shadow, 255)

        #
        # Convert to HSV color space and
        # grab the V channel
        #
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]

        #
        # Randomly choose a darkness of the shadows
        # between .45 and .75.
        #
        random_darkness = np.random.randint(45, 75) / 100.0

        #
        # modify the V channel wherever the mask is white
        #
        v_channel[mask==255] = v_channel[mask==255] * random_darkness

        #
        # Add the modifed V channel back
        #
        hsv[:, :, 2] = v_channel

        #
        # Convert back to BGR
        #
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return bgr



#
# this method returns the given image with
# a random brightness applied.
#
def random_brightness(image):
    #
    # convert to hsv.
    #
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #
    # grab the brightness channel
    #
    v = hsv[:, :, 2]

    #
    # get a random number between -100 and 99
    # to represent the change in brightness
    #
    brightness_change = np.random.randint(-100, 100)

    #
    # Apply brightness change
    #
    if brightness_change > 0:
        lim = 255 - brightness_change
        v[v > lim] = 255
        v[v <= lim] += brightness_change
    else:
        make_positive = brightness_change * -1.0
        v[v < make_positive] = 0
        v[v >= make_positive] + brightness_change

    #
    # put the altered channel back and convert
    # back to BGR
    #
    hsv[:,:,2] = v

    #
    # Convert back to BGR and return
    #
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)



def get_generator(images, batch_size):
    while True:
        #
        # Grab a random sample of size 'batch_size'
        # from the 'images' array.
        #
        batch = np.random.choice(a = images, size = batch_size)

        X = []
        y = []

        for index in range(len(batch)):
            image_data = batch[index]

            image_path = image_data.path
            if (os.path.isfile(image_path)):
                #
                # Read the image, apply augmentations
                # and add to the batch
                #
                image = cv2.imread(image_path)
                angle = image_data.angle

                if image is not None:
                    #
                    # Apply Augmentations:
                    #
                    if image_data.flipped == 1:
                        image = cv2.flip(image, 1)

                    if image_data.blur == 1:
                        image = random_blur(image)

                    if image_data.bright == 1:
                        image = random_brightness(image)

                    if image_data.shadow == 1:
                        image = random_shadow(image)

                    image = preprocess_image(image)
                    #
                    # Add to the current batch
                    #
                    X.append(image)
                    y.append(angle)


        #
        # Convert to numpy arrays
        #
        X = np.array(X)
        y = np.array(y)

        yield shuffle(X, y)
