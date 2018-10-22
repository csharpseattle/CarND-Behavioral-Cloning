#
# class BehavioralCloning
#
# A neural network model based on the NVIDIA self-driving model.
# https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
#
#
class BehavioralCloning:
    def __init__(self, path):

        from keras.models import Sequential
        from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D, BatchNormalization, Activation
        from keras.layers.convolutional import Convolution2D
        from keras.layers.pooling import MaxPooling2D
        from keras.regularizers import l2
        from keras.layers.advanced_activations import ELU
        from keras import optimizers
        from keras.callbacks import ModelCheckpoint

        self.model = Sequential()

        #
        # Normalization Output shape: (None, 128, 128, 3)
        #
        self.model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape=(128, 128, 3)))

        #
        # First Convolutional Layer.  Output shape: (None, 124, 124, 24)
        #
        self.model.add(Convolution2D(24, kernel_size=(5, 5), padding='valid', activation='relu'))

        #
        # MaxPooling, Output shape: (None, 62, 62, 24)
        #
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        #
        # Second Convolutional Layer, Output shape: (None, 58, 58, 36)
        #
        self.model.add(Convolution2D(36, kernel_size=(5, 5), padding='valid', activation='relu'))

        #
        # Max Pooling, Output shape: (None, 29, 29, 36)
        #
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        #
        # Third Convolutional Layer, Output shape: (None, 25, 25, 48)
        #
        self.model.add(Convolution2D(48, kernel_size=(5, 5), padding='valid', activation='relu'))

        #
        # Max Pooling, Output shape: (None, 12, 12, 48)
        #
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        #
        # Fourth Convolutional Layer, Output shape: (None, 10, 10, 64)
        #
        self.model.add(Convolution2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))

        #
        # Fifth Convolutional Layer, Output shape: (None, 8, 8, 64)
        #
        self.model.add(Convolution2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))

        #
        # Flattening Layer, Output shape: (None, 4096)
        #
        self.model.add(Flatten())

        #
        # Dropout -- 0.5, Output shape: (None, 4096)
        #
        self.model.add(Dropout(0.5))

        #
        # First Fully connected, Output shape: (None, 1164)
        #
        self.model.add(Dense(1164, activation='relu'))

        #
        # Dropout, Output shape: (None, 1164)
        #
        self.model.add(Dropout(0.5))

        #
        # Second Fully Connexted, Output shape: (None, 100)
        #
        self.model.add(Dense(100, activation='relu'))

        #
        # Third Fully Connected, Output shape: (None, 50)
        #
        self.model.add(Dense(50, activation='relu'))

        #
        # Fourth Fully Connected, Output shape: (None, 10)
        #
        self.model.add(Dense(10, activation='relu'))

        #
        # Fifth Fully Connected, Output shape: (None, 1)
        #
        self.model.add(Dense(1, kernel_initializer='normal'))

        #
        # Adam optimizer with learning rate of 1e-4
        #
        optimizer = optimizers.Adam(lr = 0.0001)
        self.model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        self.model.summary()

        #
        # Use the Keras ModelCheckpoint to save the model
        # after every epoch
        #
        model_checkpoint = ModelCheckpoint(path, save_best_only=True)
        self.callbacks = [model_checkpoint]


    #
    # fit()
    # Trains the model using Keras' fit_generator()
    #
    # train_generator: generator to provide batches of training data
    # valid_generator: generator to provide batches of validation data
    # training_steps: integer of training steps
    # validation_steps: integer of validation steps
    # epochs: integer of the the number of training epochs.
    #
    def fit(self, train_generator, valid_generator, training_steps, validation_steps, epochs=10):
        print("Training with {} training steps.  {} validation steps. ".format(training_steps, validation_steps))

        self.model.fit_generator(train_generator,
                            steps_per_epoch  = training_steps,
                            validation_data  = valid_generator,
                            validation_steps = validation_steps,
                            epochs           = epochs,
                            callbacks        = self.callbacks)
