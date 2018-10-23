import data
import math
import sys
from model import BehavioralCloning

BATCH_SIZE       = 256
AUGMENT_DATA     = False
FIX_DISTRIBUTION = True

if __name__ == "__main__":

    NUMBER_EPOCHS = 10

    if len(sys.argv) > 1:
        NUMBER_EPOCHS = int(sys.argv[1])

    #
    # Load image paths and steering data into memory from the
    # csv file.  Images are not actually loaded until needed
    # by the generator when making batches.
    #
    training_set, validation_set = data.load_data_sets('data/driving_log.csv')

    print("EPOCHS: {}".format(str(NUMBER_EPOCHS)))
    print("Training Set Size: {}".format(str(len(training_set))))
    print("Validation Set Size: {}".format(str(len(validation_set))))
    print("Batch Size: {}\n".format(str(BATCH_SIZE)))

    #
    # Perform data set augmentation if necessary.
    #
    if (AUGMENT_DATA):
        training_set = data.augment_dataset(training_set, FIX_DISTRIBUTION)

    #
    # Balance the distribution if called for
    #
    if (FIX_DISTRIBUTION):
        bins, bin_counts, desired_count_per_bin = data.get_bin_counts(training_set)
        training_set = data.fix_distribution(training_set, bins, bin_counts, desired_count_per_bin)

    print("Training Set Size After Augmentation and fixing Distribution: {}".format(str(len(training_set))))


    #
    # strike up a generator for the training set and another for the
    # validation set.
    #
    train_generator = data.get_generator(training_set, BATCH_SIZE)
    valid_generator = data.get_generator(validation_set, BATCH_SIZE)

    #
    # Our modified NVIDIA neural network
    #
    model = BehavioralCloning('model-{epoch:02d}.h5')

    #
    # The number of training steps per epoch is the total size
    # of the dataset divided by our batch size.
    #
    training_steps = math.ceil(len(training_set) / BATCH_SIZE)

    #
    # Similarly for valldation steps.
    #
    validation_steps = math.ceil(len(validation_set) / BATCH_SIZE)

    #
    # train!
    #
    model.fit(train_generator, valid_generator, training_steps, validation_steps, NUMBER_EPOCHS)
