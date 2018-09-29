import tensorflow as tf
import numpy as np
from models import VGG16
from input_data import DataReader
import time
import os

class Trainer:
    def __init__(self, model = None, train_file=None, test_file=None,
                    max_steps=50000, interval=20, batch_size=20, 
                    model_save_dir=None, save_prefix=None, pretrain_model=None):
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        self._train_file = train_file
        self._test_file = test_file
        self._model_save_dir = model_save_dir
        self._save_prefix = save_prefix
        self._max_steps = max_steps
        self._interval = interval
        self._batch_size = batch_size
        self._model = model
        self._sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        self._sess.run(init)
        if pretrain_model != None:
            saver = tf.train.import_meta_graph(pretrain_model+".meta")
            saver.restore(self._sess, pretrain_model)
            self._graph = tf.get_default_graph()
        else:
            self._graph = None

    def run_training(self):
        # create the model
        self._model.create_model(graph=self._graph)
        # create a saver
        saver = tf.train.Saver()
        # create a dataReader
        train_data_reader = DataReader(filename=self._train_file)
        test_data_reader = DataReader(filename=self._test_file, mode="test")


        images_placeholder = self._model.image # the input placeholder
        labels_placeholder = self._model.label #  the label placeholder
        keep_prob = self._model.keep_prob # the keep_prob placeholder
        loss = self._model.loss()
        train_op = self._model.train()
        acc = self._model.accuracy()
        out = self._model.output
        init = tf.global_variables_initializer()

        # create a session
        self._sess.run(init)
        for step in range(self._max_steps):
            start_time = time.time()
            train_images, train_labels, _ = train_data_reader.read_clip_and_label(self._batch_size)               
            self._sess.run(train_op, feed_dict={
                            images_placeholder: train_images,
                            labels_placeholder: train_labels,
                            keep_prob: 0.7
                            })
            duration = time.time() - start_time
            print('Step %d: %.3f sec' % (step, duration))

            # Save a checkpoint and evaluate the model periodically.
            if (step) % self._interval == 0 or (step + 1) == self._max_steps:
                saver.save(self._sess, os.path.join(self._model_save_dir, self._save_prefix), global_step=step)
                print('Training Data Eval:')
                now_acc, now_loss = self._sess.run(
                                [acc, loss],
                                feed_dict={images_placeholder: train_images,
                                    labels_placeholder: train_labels,
                                    keep_prob: 0.7
                                    })
                now_out = self._sess.run(out, feed_dict={images_placeholder: train_images, labels_placeholder: train_labels, keep_prob:0.5})
                print("\033[31m train out is :\033[0m",now_out[10,:])
                print ("acc: " + "{:.5f}".format(now_acc))
                print ("loss: " + "{:.5f}".format(now_loss))
                print('Validation Data Eval:')
                val_images, val_labels, _ = test_data_reader.read_clip_and_label(self._batch_size)
                now_acc, now_loss = self._sess.run(
                                [acc, loss],
                                feed_dict={images_placeholder: val_images,
                                        labels_placeholder: val_labels,
                                        keep_prob: 1
                                        })
                now_out = self._sess.run(out, feed_dict={images_placeholder: val_images, labels_placeholder: val_labels, keep_prob:1})
                print("\033[31m test out is :\033[0m",now_out[10,:])
                print ("accuracy: " + "{:.5f}".format(now_acc))
                print ("loss: " + "{:.5f}".format(now_loss))

        print("done")   

def train_c3d_model():
    # Config
    train_filename = "../list/train_list.txt"
    test_filename = "../list/test_list.txt"
    model_save_dir = "/media/storage/liweijie/c3d_models/new_try"
    save_prefix = "try"  # the format of model save name 

    my_model = VGG16()
    trainer = Trainer(model=my_model, train_file=train_filename, test_file= test_filename,
                        model_save_dir=model_save_dir, save_prefix=save_prefix, 
                        pretrain_model="/media/storage/liweijie/c3d_models/new_try/try-100")
    trainer.run_training()

if __name__ == '__main__':
    train_c3d_model()


