import tensorflow as tf
import os
import resnet_dogs_preprocessing as imagenet_preprocessing

_NUM_TRAIN_FILES = 12
_CYCLE_LENGTH = 10
_NUM_THREADS = 2
_BUFFER_LENGTH = 1024
_DEFAULT_IMAGE_SIZE = 227
_NUM_CHANNELS = 3
_NUM_SHARDS = 12

# old opt fields
# opt.dnn.name
# self.opt.dataset.log_dir_base
# self.opt.dataset.log_name
# self.opt.yuv
# self.opt.dataset.num_images_training
# self.opt.dataset.num_images_validation
# self.opt.hyper.batch_size


class ImagenetDataset:

    def __init__(self,
                 set_name='train',
                 repeat=False,
                 dnn_name='resnet',
                 dataset_log_dir_base=None,
                 dataset_log_name=None,
                 num_images_train=10800,
                 num_images_validation=1200,
                 num_images_test=8580,
                 hyper_batch_size=64,
                 yuv=False,
                 crop=True):
        self.set_name = set_name
        # self.num_total_images
        self.dnn_name = dnn_name
        self.dataset_log_dir_base = dataset_log_dir_base
        self.dataset_log_name = dataset_log_name
        self.num_images_train = num_images_train
        self.num_images_validation = num_images_validation  # dog dataset
        self.num_images_test = num_images_test
        self.hyper_batch_size = hyper_batch_size
        self.yuv = yuv
        self.crop = crop
        self.handle = self.create_dataset(repeat)
        self.num_total_images = None

    def create_dataset(self, repeat=False):

        def _parse_function(example_serialized):
            """Parses an Example proto containing a training example of an image.

            The output of the build_imagenet_dogs_data.py image preprocessing script is a dataset
            containing serialized Example protocol buffers. Each Example proto contains
            the following fields (values are included as examples):

              image/height: 462
              image/width: 581
              image/colorspace: 'RGB'
              image/channels: 3
              image/class/label: 615
              image/class/synset: 'n03623198'
              image/class/text: 'knee pad'
              image/object/bbox/xmin: 0.1
              image/object/bbox/xmax: 0.9
              image/object/bbox/ymin: 0.2
              image/object/bbox/ymax: 0.6
              image/object/bbox/label: 615
              image/format: 'JPEG'
              image/filename: 'ILSVRC2012_val_00041207.JPEG'
              image/encoded: <JPEG encoded string>

            Args:
              example_serialized: scalar Tensor tf.string containing a serialized
                Example protocol buffer.

            Returns:
              image_buffer: Tensor tf.string containing the contents of a JPEG file.
              label: Tensor tf.int32 containing the label.
              bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
                where each coordinate is [0, 1) and the coordinates are arranged as
                [ymin, xmin, ymax, xmax].
            """
            # Dense features in Example proto.
            feature_map = {
                'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                                    default_value=''),
                'image/class/label': tf.FixedLenFeature([], dtype=tf.int64,
                                                        default_value=-1),
                'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                                       default_value=''),
            }
            sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
            # Sparse features in Example proto.
            feature_map.update(
                {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                             'image/object/bbox/ymin',
                                             'image/object/bbox/xmax',
                                             'image/object/bbox/ymax']})

            features = tf.parse_single_example(example_serialized, feature_map)
            label = tf.cast(features['image/class/label'], dtype=tf.int32)

            xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
            ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
            xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
            ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

            # Note that we impose an ordering of (y, x) just to make life difficult.
            bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

            # Force the variable number of bounding boxes into the shape
            # [1, num_boxes, coords].
            bbox = tf.expand_dims(bbox, 0)
            bbox = tf.transpose(bbox, [0, 2, 1])

            if self.dnn_name == 'resnet':
                image = imagenet_preprocessing.preprocess_image(
                    image_buffer=features['image/encoded'],
                    bbox=bbox,
                    label=label,
                    output_height=_DEFAULT_IMAGE_SIZE,
                    output_width=_DEFAULT_IMAGE_SIZE,
                    num_channels=_NUM_CHANNELS,
                    crop=self.crop)

            elif self.dnn_name == 'inception':
                image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
                # Crop the central region of the image with an area containing 87.5% of
                # the original image.
                image = tf.image.central_crop(image, central_fraction=0.875)

                # Resize the image to the original height and width.
                image = tf.expand_dims(image, 0)
                image = tf.image.resize_bilinear(image, [299, 299],
                                                 align_corners=False)
                image = tf.squeeze(image, [0])

                image = tf.subtract(image/256, 0.5)
                image = tf.multiply(image, 2.0)

            elif self.dnn_name == 'AlexNet':
                if self.yuv == False:
                    with tf.device("/device:cpu:0"):
                        image = imagenet_preprocessing.preprocess_image(
                            image_buffer=features['image/encoded'],
                            bbox=bbox,
                            label=label,
                            output_height=227,
                            output_width=227,
                            num_channels=_NUM_CHANNELS,
                            is_training=(self.set_name == 'train'))
                else:
                    with tf.device("/device:cpu:0"):
                        image = imagenet_preprocessing.preprocess_image_Yuv(
                            image_buffer=features['image/encoded'],
                            bbox=bbox,
                            label=label,
                            output_height=227,
                            output_width=227,
                            num_channels=_NUM_CHANNELS,
                            is_training=(self.set_name == 'train'))

            return image, label

        tfrecords_path = os.path.join(self.dataset_log_dir_base,  self.dataset_log_name)
        # filenames = self.get_filenames(self.set_name, tfrecords_path)

        if self.set_name == 'train':
            files = tf.data.Dataset.list_files(tfrecords_path + '/train-*')
        elif self.set_name == 'validation':
            files = tf.data.Dataset.list_files(tfrecords_path + '/validation-*')
        elif self.set_name == 'test':
            files = tf.data.Dataset.list_files(tfrecords_path + '/test-*')
        else:
            raise ValueError("This TFRecord does not exists")

        dataset = files.interleave(
            tf.data.TFRecordDataset, cycle_length=8,
            num_parallel_calls=8)

        if self.set_name == 'train':
            # Shuffle the input files (TODO: ask Xavier why we need this when we already have shuffled data)
            dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)
            self.num_total_images = self.num_images_train
        elif self.set_name == 'validation':
            self.num_total_images = self.num_images_validation
        elif self.set_name == 'test':
            self.num_total_images = self.num_images_test
        else:
            raise ValueError("This split does not exist")
        print('Number of total images', self.num_total_images)

        if repeat:
            dataset = dataset.repeat()  # Repeat the input indefinitely.

        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            map_func=_parse_function,
            batch_size=self.hyper_batch_size,
            num_parallel_batches=8))

        dataset = dataset.prefetch(buffer_size=2)

        return dataset  # dataset.batch(self.opt.hyper.batch_size)
