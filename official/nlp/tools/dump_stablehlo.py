"""Utility to dump StableHLO for an NLP model training step.

This script builds a model from an experiment config and runs a single
training step with dummy data under XLA compilation. The StableHLO text for
that compiled training step is written to the specified output file.

Example:
```
python -m official.nlp.tools.dump_stablehlo \
    --experiment=bert/sentence_prediction \
    --output=/tmp/bert_train.stablehlo
```
"""
from absl import app
from absl import flags
import tensorflow as tf, tf_keras

from official.core import exp_factory
from official.core import task_factory
from official.modeling import optimization
from official.common import registry_imports  # pylint: disable=unused-import

FLAGS = flags.FLAGS

flags.DEFINE_string('experiment', None, 'Experiment config name.')
flags.DEFINE_string('output', 'stablehlo.txt', 'File to write StableHLO text.')
flags.DEFINE_integer('batch_size', 1, 'Global batch size for dummy inputs.')


def main(_):
  params = exp_factory.get_exp_config(FLAGS.experiment)
  params.task.train_data.input_path = 'dummy'
  params.task.train_data.global_batch_size = FLAGS.batch_size
  if getattr(params.task.model, 'num_classes', 0) <= 0:
    params.task.model.num_classes = 2
  params.trainer.train_steps = 1
  params.trainer.optimizer_config.learning_rate.type = 'constant'
  params.trainer.optimizer_config.learning_rate.constant.learning_rate = 0.0
  params.trainer.optimizer_config.warmup.type = None

  strategy = tf.distribute.get_strategy()
  with strategy.scope():
    task = task_factory.get_task(params.task)
    model = task.build_model()
    opt_factory = optimization.OptimizerFactory(params.trainer.optimizer_config)
    learning_rate = opt_factory.build_learning_rate()
    optimizer = opt_factory.build_optimizer(learning_rate)
    metrics = task.build_metrics(training=True)

  @tf.function(jit_compile=True)
  def train_step(inputs):
    return task.train_step(inputs,
                           model=model,
                           optimizer=optimizer,
                           metrics=metrics)

  seq_length = params.task.train_data.seq_length
  dummy_ids = tf.zeros([FLAGS.batch_size, seq_length], tf.int32)
  sample = {
      'input_word_ids': dummy_ids,
      'input_mask': dummy_ids,
      'input_type_ids': dummy_ids,
  }
  label_field = getattr(params.task.train_data, 'label_field', 'label_ids')
  if params.task.model.num_classes == 1:
    sample[label_field] = tf.zeros([FLAGS.batch_size], tf.float32)
  else:
    sample[label_field] = tf.zeros([FLAGS.batch_size], tf.int32)

  train_step(sample)
  hlo_text = train_step.experimental_get_compiler_ir(sample)(stage='stablehlo')

  with tf.io.gfile.GFile(FLAGS.output, 'w') as f:
    f.write(hlo_text)
  tf.print('StableHLO written to', FLAGS.output)


if __name__ == '__main__':
  flags.mark_flag_as_required('experiment')
  app.run(main)
