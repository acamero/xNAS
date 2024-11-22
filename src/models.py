import tensorflow as tf
 

def fcn_model(train_images,
              train_labels,
              num_classes,
              dropout_rate=0.2,
              **kwargs):
    input = tf.keras.layers.Input(shape=train_images.shape[1:])
    # A convolution block
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same")(input)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Fully connected layer 1
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Fully connected layer 2
    x = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=1, strides=1)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)    
    # FF
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)       
    model = tf.keras.Model(inputs=input, outputs=predictions)
    
    model.compile(
            optimizer='adam', 
            loss=tf.keras.losses.categorical_crossentropy, 
            metrics=['accuracy'])
    history = model.fit(train_images, train_labels, **kwargs)    
    
    return model  



class ModelLoader():
    def __init__(self, 
                 train_images=None,
                 train_labels=None,
                 num_classes=None,
                 **kwargs):
        self.train_images = train_images
        self.train_labels = train_labels
        self.num_classes = num_classes
        self.kwargs = kwargs
        self.models = dict()
        self.models['fcn'] = fcn_model

        
    def get_model(self, name):
        assert name in self.models
        model = self.models[name](self.train_images, self.train_labels, self.num_classes, **self.kwargs)
        return model


