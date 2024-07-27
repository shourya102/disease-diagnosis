import os

from keras.applications.resnet import ResNet50
from keras.layers.core import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 32


def generator(path):
    classes = os.listdir(path)
    batches = ImageDataGenerator().flow_from_directory(path, target_size=(224, 224), classes=classes,
                                                       batch_size=BATCH_SIZE, class_mode='categorical')
    return batches


def fine_tune_res50(train_path, val_path, model_name, loss, classes, epochs):
    train_generator = generator(train_path)
    val_generator = generator(val_path)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(len(classes), activation='sigmoid'))
    print(model.summary())
    model.compile(optimizer=Adam(),
                  loss=loss,
                  metrics=['accuracy'])
    history = model.fit(train_generator,
                        epochs=epochs,
                        steps_per_epoch=int(train_generator.samples / BATCH_SIZE),
                        validation_data=val_generator,
                        validation_steps=int(val_generator.samples / BATCH_SIZE)
                        )
    model.save(model_name)
    print(f'Model saved at {model_name}')


if __name__ == '__main__':
    train_path = 'datasets/brain_disease_dataset/train'
    val_path = 'datasets/brain_disease_dataset/val'
    model_name = 'brain.h5'
    loss = 'categorical_crossentropy'
    classes = os.listdir(train_path)
    epochs = 10
    fine_tune_res50(train_path, val_path, model_name, loss, classes, epochs)
