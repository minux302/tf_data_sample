import cv2
from pathlib import Path
from math import pi

import tensorflow as tf
from tensorflow.image import ResizeMethod
from tensorflow.contrib.image import rotate


class Dataset():
    def __init__(self, data_dir, batch_size, image_size):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.image_size = image_size

        self._build_pipeline()

    def _imread_fn(self, image_path):
        image = tf.image.decode_jpeg(tf.read_file(image_path))
        image = tf.image.resize_images(image,
                                       tf.cast(self.image_size, tf.int32),
                                       ResizeMethod.BICUBIC)
        return tf.cast(image, tf.float32)

    def _augment_fn(self, image):
        image = tf.image.random_flip_left_right(image)
        image = rotate(image, tf.random_uniform((), minval=-1 / 4, maxval=1 / 4) * pi)
        return image

    def _build_pipeline(self):
        image_list = [str(image_path) for image_path in self.data_dir.glob('**/*.jpg')]
        dataset = tf.data.Dataset.from_tensor_slices(image_list)
        dataset = dataset.map(self._imread_fn)
        dataset = dataset.map(self._augment_fn)

        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(self.batch_size)

        iterator = dataset.make_initializable_iterator()
        self.initializer = iterator.initializer
        self.batch = iterator.get_next()


def main():
    tf.reset_default_graph()
    dataset = Dataset(data_dir='data',
                      batch_size=32,
                      image_size=(224, 224))

    save_dir = Path('save')
    if not save_dir.exists():
        save_dir.mkdir()

    counter = 0
    with tf.Session() as sess:
        for epoch in range(2):
            sess.run(dataset.initializer)
            while True:
                try:
                    image = sess.run(dataset.batch)
                    cv2.imwrite(str(save_dir / '{}.jpg'.format(counter)), image[0])
                    counter += 1
                except tf.errors.OutOfRangeError:
                    break
    print('finished')


if __name__ == '__main__':
    main()
