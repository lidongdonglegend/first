import os
import re
import tensorflow as tf

# look for the tfrecord, save the info as a dict
tfrecord_list = os.listdir('train')
vid_labels_dict = {}
for tfrecord in tfrecord_list:
    if re.match("^train.*?tfrecord",tfrecord):
        frame_lvl_record = "train/{}".format(tfrecord)
        sub_vid_ids = []
        sub_labels = []
        for example in tf.python_io.tf_record_iterator(frame_lvl_record):
            tf_example = tf.train.Example.FromString(example)
            sub_vid_ids.append(tf_example.features.feature['id']
                           .bytes_list.value[0].decode(encoding='UTF-8'))
            sub_labels.append(tf_example.features.feature['labels'].int64_list.value)
        vid_labels_dict[tfrecord] = dict(zip(sub_vid_ids,sub_labels))

# save the vid_labels_dict
with open('vid_info.txt','a+',encoding='utf-8') as f:
    f.write(str(vid_labels_dict))
