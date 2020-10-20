# ==============================================================================
# This script is used to genrate tfrecord file for training.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

# pylint: disable=g-bad-import-order
import six
from absl import app as absl_app
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.transformer.utils import tokenizer
from official.transformer.utils.string_utils import split_zhcn
from official.utils.flags import core as flags_core


# Data sources for training/evaluating the transformer translation model.
# If any of the training sources are changed, then either:
#   1) use the flag `--search` to find the best min count or
#   2) update the _TRAIN_DATA_MIN_COUNT constant.
# min_count is the minimum number of times a token must appear in the data
# before it is added to the vocabulary. "Best min count" refers to the value
# that generates a vocabulary set that is closest in size to _TARGET_VOCAB_SIZE.

# Use pre-defined minimum count to generate subtoken vocabulary.
_TRAIN_DATA_MIN_COUNT = 6
_TARGET_VOCAB_SIZE = 32768  # Number of subtokens in the vocabulary list.
_TARGET_THRESHOLD = 327  # Accept vocabulary if size is within this threshold
_PREFIX = "sub271"
_TRAIN_TAG = "train"
_EVAL_TAG = "dev"
_TRAIN_SHARDS = 100
_EVAL_SHARDS = 1


###############################################################################
# Extraction functions
###############################################################################
def get_raw_files(raw_dir, data_source):
  """Return raw files from source. Downloads/extracts if needed.

  Args:
    raw_dir: string directory to store raw files
    data_source: dictionary
  """
  raw_files = {
      "inputs": [],
      "targets": [],
  }  # keys
  for d in data_source:
    raw_files["inputs"].append(os.path.join(FLAGS.raw_dir, d["input"]))
    raw_files["targets"].append(os.path.join(FLAGS.raw_dir, d["target"]))
  return raw_files


def txt_line_iterator(path):
  """Iterate through lines of file."""
  with tf.gfile.Open(path) as f:
    for line in f:
      yield line.strip()


def compile_files(raw_dir, raw_files, tag):
  """Compile raw files into a single file for each language.

  Args:
    raw_dir: Directory containing downloaded raw files.
    raw_files: Dict containing filenames of input and target data.
      {"inputs": list of files containing data in input language
       "targets": list of files containing corresponding data in target language
      }
    tag: String to append to the compiled filename.

  Returns:
    Full path of compiled input and target files.
  """
  tf.logging.info("Compiling files with tag %s." % tag)
  filename = "%s-%s" % (_PREFIX, tag)
  input_compiled_file = os.path.join(raw_dir, filename + ".lang1")
  target_compiled_file = os.path.join(raw_dir, filename + ".lang2")

  with tf.gfile.Open(input_compiled_file, mode="w") as input_writer:
    with tf.gfile.Open(target_compiled_file, mode="w") as target_writer:
      for i in range(len(raw_files["inputs"])):
        input_file = raw_files["inputs"][i]
        target_file = raw_files["targets"][i]

        tf.logging.info("Reading files %s and %s." % (input_file, target_file))
        write_file(input_writer, input_file)
        write_file(target_writer, target_file)
  return input_compiled_file, target_compiled_file


def write_file(writer, filename):
  """Write all of lines from file using the writer."""
  for line in txt_line_iterator(filename):
    writer.write(line)
    writer.write("\n")


###############################################################################
# Data preprocessing
###############################################################################
def encode_and_save_files(
    subtokenizer, data_dir, raw_files, tag, total_shards, src_lang, tgt_lang):
  """Save data from files as encoded Examples in TFrecord format.

  Args:
    subtokenizer: Subtokenizer object that will be used to encode the strings.
    data_dir: The directory in which to write the examples
    raw_files: A tuple of (input, target) data files. Each line in the input and
      the corresponding line in target file will be saved in a tf.Example.
    tag: String that will be added onto the file names.
    total_shards: Number of files to divide the data into.

  Returns:
    List of all files produced.
  """
  # Create a file for each shard.
  filepaths = [shard_filename(data_dir, tag, n + 1, total_shards)
               for n in range(total_shards)]

  if all_exist(filepaths):
    tf.logging.info("Files with tag %s already exist." % tag)
    return filepaths

  tf.logging.info("Saving files with tag %s." % tag)
  input_file = raw_files[0]
  target_file = raw_files[1]

  # Write examples to each shard in round robin order.
  tmp_filepaths = [fname + ".incomplete" for fname in filepaths]
  writers = [tf.python_io.TFRecordWriter(fname) for fname in tmp_filepaths]
  counter, shard = 0, 0
  for counter, (input_line, target_line) in enumerate(zip(
      txt_line_iterator(input_file), txt_line_iterator(target_file))):
    if counter > 0 and counter % 100000 == 0:
      tf.logging.info("\tSaving case %d." % counter)

    ### Add DIY input format.
    src_items = input_line.split('\1')
    cur_src_line = src_items[1] if len(src_items) > 1 else src_items[0]
    if src_lang == 'zh':
      src_list = subtokenizer.encode(split_zhcn(cur_src_line), add_eos=True)
    else:
      src_list = subtokenizer.encode(cur_src_line, add_eos=True)
    tgt_items = target_line.split('\1')
    cur_tgt_line = tgt_items[1] if len(tgt_items) > 1 else tgt_items[0]
    if tgt_lang == 'zh':
      tgt_list = subtokenizer.encode(split_zhcn(cur_tgt_line), add_eos=True)
    else:
      tgt_list = subtokenizer.encode(cur_tgt_line, add_eos=True)
    example = dict_to_example(
      {"inputs": src_list,
       "targets": tgt_list})
    if counter < 3:
      tf.logging.info("counter: %s" % counter)
      tf.logging.info("inputs: %s" % cur_src_line)
      tf.logging.info("targets: %s" % cur_tgt_line)
      tf.logging.info("inputs ids: %s" % ' '.join([str(id) for id in src_list]))
      tf.logging.info("targets ids: %s" % ' '.join([str(id) for id in tgt_list]))

    writers[shard].write(example.SerializeToString())
    shard = (shard + 1) % total_shards
  for writer in writers:
    writer.close()

  for tmp_name, final_name in zip(tmp_filepaths, filepaths):
    tf.gfile.Rename(tmp_name, final_name)

  tf.logging.info("Saved %d Examples", counter + 1)
  return filepaths


def shard_filename(path, tag, shard_num, total_shards):
  """Create filename for data shard."""
  return os.path.join(
      path, "%s-%s-%.5d-of-%.5d" % (_PREFIX, tag, shard_num, total_shards))


def shuffle_records(fname):
  """Shuffle records in a single file."""
  tf.logging.info("Shuffling records in file %s" % fname)

  # Rename file prior to shuffling
  tmp_fname = fname + ".unshuffled"
  tf.gfile.Rename(fname, tmp_fname)

  reader = tf.io.tf_record_iterator(tmp_fname)
  records = []
  for record in reader:
    records.append(record)
    if len(records) % 100000 == 0:
      tf.logging.info("\tRead: %d", len(records))

  random.shuffle(records)

  # Write shuffled records to original file name
  with tf.python_io.TFRecordWriter(fname) as w:
    for count, record in enumerate(records):
      w.write(record)
      if count > 0 and count % 100000 == 0:
        tf.logging.info("\tWriting record: %d" % count)

  tf.gfile.Remove(tmp_fname)


def dict_to_example(dictionary):
  """Converts a dictionary of string->int to a tf.Example."""
  features = {}
  for k, v in six.iteritems(dictionary):
    features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
  return tf.train.Example(features=tf.train.Features(feature=features))


def all_exist(filepaths):
  """Returns true if all files in the list exist."""
  for fname in filepaths:
    if not tf.gfile.Exists(fname):
      return False
  return True


def make_dir(path):
  if not tf.gfile.Exists(path):
    tf.logging.info("Creating directory %s" % path)
    tf.gfile.MakeDirs(path)


def main(unused_argv):
  train_data_sources = [
      {
          "input": "sub_iqiyi_%s%s_train.%s" % (FLAGS.src_lang, FLAGS.tgt_lang, FLAGS.src_lang),
          "target": "sub_iqiyi_%s%s_train.%s" % (FLAGS.src_lang, FLAGS.tgt_lang, FLAGS.tgt_lang),
      },
  ]
  eval_data_sources = [
      {
          "input": "sub_iqiyi_%s%s_dev.%s" % (FLAGS.src_lang, FLAGS.tgt_lang, FLAGS.src_lang),
          "target": "sub_iqiyi_%s%s_dev.%s" % (FLAGS.src_lang, FLAGS.tgt_lang, FLAGS.tgt_lang),
      }
  ]
  vocab_file = os.path.join(FLAGS.data_dir, "vocab.%s%s" % (FLAGS.src_lang, FLAGS.tgt_lang))
  make_dir(FLAGS.raw_dir)
  make_dir(FLAGS.data_dir)

  # Get paths of download/extracted training and evaluation files.
  tf.logging.info("Step 1/4: Downloading data from source")
  train_files = get_raw_files(FLAGS.raw_dir, train_data_sources)
  eval_files = get_raw_files(FLAGS.raw_dir, eval_data_sources)

  # Create subtokenizer based on the training files.
  tf.logging.info("Step 2/4: Creating subtokenizer and building vocabulary")
  train_files_flat = train_files["inputs"] + train_files["targets"]
  train_files_flat_4v = []
  for train_file in train_files_flat:
    tf.logging.info(train_file)
    train_files_flat_4v.append(train_file + '.4v')
  subtokenizer = tokenizer.Subtokenizer.init_from_files(
      vocab_file, train_files_flat_4v, _TARGET_VOCAB_SIZE, _TARGET_THRESHOLD,
      min_count=None if FLAGS.search else _TRAIN_DATA_MIN_COUNT, file_byte_limit=1e6)

  tf.logging.info("Step 3/4: Compiling training and evaluation data")
  compiled_train_files = compile_files(FLAGS.raw_dir, train_files, _TRAIN_TAG)
  compiled_eval_files = compile_files(FLAGS.raw_dir, eval_files, _EVAL_TAG)

  # Tokenize and save data as Examples in the TFRecord format.
  tf.logging.info("Step 4/4: Preprocessing and saving data")
  train_tfrecord_files = encode_and_save_files(
      subtokenizer, FLAGS.data_dir, compiled_train_files, _TRAIN_TAG,
      _TRAIN_SHARDS, FLAGS.src_lang, FLAGS.tgt_lang)
  encode_and_save_files(
      subtokenizer, FLAGS.data_dir, compiled_eval_files, _EVAL_TAG,
      _EVAL_SHARDS, FLAGS.src_lang, FLAGS.tgt_lang)

  for fname in train_tfrecord_files:
    shuffle_records(fname)


def define_data_download_flags():
  """Add flags specifying data download arguments."""
  flags.DEFINE_string(
      name="data_dir", short_name="dd", default="/tmp/translate_ende",
      help=flags_core.help_wrap(
          "Directory for where the translate_ende_wmt32k dataset is saved."))
  flags.DEFINE_string(
      name="raw_dir", short_name="rd", default="/tmp/translate_ende_raw",
      help=flags_core.help_wrap(
          "Path where the raw data will be downloaded and extracted."))
  flags.DEFINE_bool(
      name="search", default=False,
      help=flags_core.help_wrap(
          "If set, use binary search to find the vocabulary set with size"
          "closest to the target size (%d)." % _TARGET_VOCAB_SIZE))
  flags.DEFINE_string(
      name="src_lang", short_name="sl", default="zh",
      help=flags_core.help_wrap(
          "source language used to read corresponding files."))
  flags.DEFINE_string(
      name="tgt_lang", short_name="tl", default="en",
      help=flags_core.help_wrap(
          "target language used to read corresponding files."))


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  define_data_download_flags()
  FLAGS = flags.FLAGS
  absl_app.run(main)
