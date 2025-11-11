import argparse, os, json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers, callbacks

def build_model(num_classes, img_size=(224,224,3), learning_rate=1e-4):
    base = MobileNetV2(include_top=False, weights='imagenet', input_shape=img_size)
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base.input, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main(args):
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    img_size = (args.img_size, args.img_size)
    batch = args.batch_size

    train_gen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
    val_gen = ImageDataGenerator(rescale=1./255)

    train_flow = train_gen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch, class_mode='categorical')
    val_flow = val_gen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch, class_mode='categorical')

    num_classes = len(train_flow.class_indices)
    print('Detected classes:', train_flow.class_indices)

    model = build_model(num_classes, img_size=img_size+(3,), learning_rate=args.lr)
    callbacks_list = [
        callbacks.ModelCheckpoint(os.path.join('models','rice_model.h5'), save_best_only=True, monitor='val_loss'),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
        callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    ]
    history = model.fit(train_flow, validation_data=val_flow, epochs=args.epochs, callbacks=callbacks_list)

    os.makedirs('models', exist_ok=True)
    model.save('models/rice_model.h5')
    with open('models/class_indices.json','w') as f:
        json.dump(train_flow.class_indices, f)
    print('Training finished. Model saved to models/rice_model.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset', help='Dataset directory with train/val subfolders')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    main(args)
