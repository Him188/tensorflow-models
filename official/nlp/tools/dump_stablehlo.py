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

  dummy_ds = task.build_inputs(params.task.train_data)
  sample = next(iter(dummy_ds))

  train_step(sample)
  hlo_text = train_step.experimental_get_compiler_ir(sample)(stage='stablehlo')

  with tf.io.gfile.GFile(FLAGS.output, 'w') as f:
    f.write(hlo_text)
  tf.print('StableHLO written to', FLAGS.output)


if __name__ == '__main__':
  flags.mark_flag_as_required('experiment')
  app.run(main)
