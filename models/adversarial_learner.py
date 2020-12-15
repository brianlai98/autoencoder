import os
import sys
import time
from itertools import count
import math
import numpy as np
import tensorflow as tf
from keras.utils.generic_utils import Progbar
from .nets import recover_net, generator_net
from .utils.loss_utils import charbonnier_loss, train_op
from data.davis2016_data_utils import Davis2016Reader

class AdversarialLearner(object):
    def __init__(self):
        pass

    def load_training_data(self):
        with tf.name_scope("data_loading"):
            if self.config.dataset== 'DAVIS2016':
                reader = Davis2016Reader(self.config.root_dir,
                                     max_temporal_len=self.config.max_temporal_len,
                                     min_temporal_len=self.config.min_temporal_len,
                                     num_threads=self.config.num_threads)

                train_batch, train_iter = reader.image_inputs(batch_size=self.config.batch_size,
                                                              train_crop=self.config.train_crop,
                                                              partition=self.config.train_partition)

                val_batch, val_iter = reader.test_inputs(batch_size=self.config.batch_size,
                                                         t_len=self.config.test_temporal_shift,
                                                         test_crop=self.config.test_crop,
                                                         partition='val')

        self.num_samples_val = reader.val_samples
        return train_batch, val_batch, train_iter, val_iter

    def build_train_graph(self):
        is_training_ph = tf.placeholder(tf.bool, shape=(), name="is_training")

        train_batch, val_batch, train_iter, val_iter = self.load_training_data()

        current_batch = tf.cond(is_training_ph, lambda: train_batch,
                                lambda: val_batch)

        image_batch, images_2_batch = current_batch[0], current_batch[1]


        # Reshape everything to desired image size
        image_batch = tf.image.resize_images(image_batch, [self.config.img_height,
                                                           self.config.img_width])


        with tf.name_scope("MaskNet") as scope:
            # This is the generator network
            generated_masks = generator_net(images=image_batch,
                                       training = is_training_ph,
                                       scope=scope,
                                       reuse=False)


        # Define now all training losses.

        losses = {}


        # Generator loss is the quality of flow reconstruction.
        generator_loss = tf.reduce_mean(tf.abs(generated_masks - image_batch))

        losses['generator'] = generator_loss

        with tf.name_scope("train_op"):
            self.global_step = tf.Variable(0,
                                           name='global_step',
                                           trainable=False)
            generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               'MaskNet')

            optimizer  = tf.train.AdamOptimizer(learning_rate=1e-4,
                                             beta1=self.config.beta1, epsilon=1e-8)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            # The can_flag puts some noise into the gradients if the latter are vanishing.
            # This usually happen when the generator encounters the local minimum of
            # masking everything or masking nothing.
            self.train_generator_op, self.generator_var_grads = train_op(loss=losses['generator'],
                                               var_list=generator_vars,
                                               optimizer=optimizer,
                                               gradient_clip_value=1000.,
                                               can_change=True)

            self.train_generator_op = tf.group([self.train_generator_op, update_ops])

            self.incr_global_step = tf.assign(self.global_step,
                                              self.global_step+1)
        self.iterators = [train_iter, val_iter]
        self.image_batch = image_batch
        self.losses = losses
        self.generated_masks = generated_masks
        
        self.is_training = is_training_ph
        self.train_steps_per_epoch = \
            int(math.ceil(self.config.num_samples_train/self.config.batch_size))
        self.val_steps_per_epoch = int(np.ceil(float(self.num_samples_val) / self.config.batch_size))


    def collect_summaries(self):
        """Collects all summaries to be shown in the tensorboard"""
        for key, value in self.losses.items():
            tf.summary.scalar(key, value, collections=['step_sum'])

        tf.summary.image("input_image", self.image_batch, max_outputs=1,
                         collections=['step_sum'])


        for grad, var in self.generator_var_grads:
            tf.summary.histogram(var.op.name + "/gradients", grad,
                                 collections=["step_sum"])

        self.step_sum = tf.summary.merge(tf.get_collection('step_sum'))
        #######################
        # VALIDATION ERROR SUM#
        #######################
        # self.val_sum = tf.summary.merge(tf.get_collection('validation_summary'))

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to {}/model-{}".format(checkpoint_dir,
                                                             step))
        if step == 'best':
            self.saver.save(sess,
                        os.path.join(checkpoint_dir, model_name + '.best'))
        else:
            self.saver.save(sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def train(self, config):
        """High level train function.
        Args:
            config: Configuration dictionary
        Returns:
            None
        """
        self.config = config
        self.build_train_graph()
        self.collect_summaries()
        self.min_val_iou = -1.0e12 # Initialize to large value
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                        for v in tf.trainable_variables()])
        self.saver = tf.train.Saver([var for var in \
            tf.trainable_variables()] + [self.global_step], max_to_keep=40)

        sv = tf.train.Supervisor(logdir=config.checkpoint_dir,
                                 save_summaries_secs=0,
                                 saver=None)

        with sv.managed_session() as sess:
            print("Number of params: {}".format(sess.run(parameter_count)))

            if config.resume_train:
                if os.path.isfile(self.config.full_model_ckpt + ".index"):
                    checkpoint = self.config.full_model_ckpt
                elif os.path.isdir(self.config.checkpoint_dir):
                    checkpoint = tf.train.latest_checkpoint(
                                                self.config.checkpoint_dir)
                assert checkpoint, "Found no checkpoint to resume training!"
                self.saver.restore(sess, checkpoint)
                print("Resumed training from model {}".format(checkpoint))
            else:
                # Better to initialize form a recover pretrained on simulated datasets.
                # This can be downloaded from the project page.
                print("No recover checkpoint found! Train Recover from Scratch")

            progbar = Progbar(target=self.train_steps_per_epoch)

            for it in self.iterators:
                sess.run(it.initializer)

            iters_rec = self.config.iters_rec
            iters_gen = self.config.iters_gen

            print("-------------------------------------")
            print("Training {} Recover and {} Generator".format(iters_rec, iters_gen))
            print("-------------------------------------")

            sum_iters = iters_rec + iters_gen

            for step in count(start=1):
                if sv.should_stop():
                    break
                start_time = time.time()
                fetches = {"global_step": self.global_step}

                if step % sum_iters == 0:
                    # Global step increased after every cycle
                    fetches.update({"incr_global_step": self.incr_global_step})

                
                fetches["train_op"] = self.train_generator_op

                if (step % config.summary_freq ==0):
                    fetches["loss_generator"] = self.losses['generator']
                    fetches["summary"] = self.step_sum

                results = sess.run(fetches,
                                   feed_dict={ self.is_training : True })

                progbar.update(step % self.train_steps_per_epoch)
                gs = results["global_step"]

                if step % config.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil( step /self.train_steps_per_epoch)
                    train_step = step - (train_epoch - 1) * self.train_steps_per_epoch
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it loss_generator: %4.4f " \
                       % (train_epoch, train_step, self.train_steps_per_epoch, \
                                time.time() - start_time, \
                          results["loss_generator"]))

                if step % self.train_steps_per_epoch == 0:
                    # This differ from the last when resuming training
                    train_epoch = int(step / self.train_steps_per_epoch)
                    progbar = Progbar(target=self.train_steps_per_epoch)
                    self.epoch_end_callback(sess, sv, train_epoch)
                    if (train_epoch == self.config.max_epochs):
                        print("-------------------------------")
                        print("Training completed successfully")
                        print("-------------------------------")
                        break

    def epoch_end_callback(self, sess, sv, epoch_num):
        # Evaluate val loss
        # Log to Tensorflow board
        val_sum = sess.run(self.val_sum)

        sv.summary_writer.add_summary(val_sum, epoch_num)

        print("Epoch [{}] Validation IoU: {}".format(
            epoch_num, validation_iou))
        if epoch_num % self.config.save_freq == 0:
            self.save(sess, self.config.checkpoint_dir, epoch_num)

    def build_test_graph(self):
        """This graph will be used for testing. In particular, it will
           compute the loss on a testing set, or some other utilities.
        """
        with tf.name_scope("data_loading"):
            if self.config.dataset== 'DAVIS2016':
                reader = Davis2016Reader(self.config.root_dir, num_threads=1)
                test_batch, test_iter = reader.test_inputs(batch_size=self.config.batch_size,
                                                      t_len=self.config.test_temporal_shift,
                                                      with_fname=True,
                                                      test_crop=self.config.test_crop,
                                                      partition=self.config.test_partition)
            else:
                raise IOError("Dataset should be DAVIS2016 / FBMS / SEGTRACK")

            image_batch, images_2_batch, gt_mask_batch, fname_batch = test_batch[0], \
                                            test_batch[1], test_batch[2], test_batch[3]

        # Reshape everything
        image_batch = tf.image.resize_images(image_batch, [self.config.img_height,
                                                           self.config.img_width])


        with tf.name_scope("MaskNet") as scope:
            generated_masks = generator_net(images=image_batch,
                                       training=False,
                                       scope=scope,
                                       reuse=False)
        self.input_image = image_batch
        self.fname_batch = fname_batch
        self.generated_masks = generated_masks
        self.test_iterator = test_iter

    def setup_inference(self, config, aug_test=False):
        """Sets up the inference graph.
        Args:
            config: config dictionary.
        """
        self.config = config
        self.aug_test = aug_test
        self.build_test_graph()

    def inference(self, sess):
        """Outputs a dictionary with the results of the required operations.
        Args:
            sess: current session
        Returns:
            results: dictionary with output of testing operations.
        """
        
        fetches = {'input_image': self.input_image, 'img_fname': self.fname_batch}

        results = sess.run(fetches)

        return results
