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


def cnn_model(train_images,
              train_labels,
              num_classes,
              **kwargs):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=train_images.shape[1:]))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))    # num_classes = 10
    model.compile(
            optimizer='adam', 
            loss=tf.keras.losses.categorical_crossentropy, 
            metrics=['accuracy'])
    history = model.fit(train_images, train_labels, **kwargs)

    return model
    

class ModelLoader():
    models = {
        'fcn': fcn_model,
        'cnn': cnn_model
    }

    def __init__(self, 
                 train_images=None,
                 train_labels=None,
                 num_classes=None,
                 **kwargs):
        self.train_images = train_images
        self.train_labels = train_labels
        self.num_classes = num_classes
        self.kwargs = kwargs
     
        
    def get_model(self, name):
        if name.endswith(".keras"):
            print("Loading saved model", name)
            return tf.keras.models.load_model(name)
        
        assert name in self.models
        model = self.models[name](self.train_images, self.train_labels, self.num_classes, **self.kwargs)
        return model


    @staticmethod
    def available_models():
        return list(ModelLoader.models.keys())
