"""Utility to dump StableHLO for an NLP model training step.

This script builds a model from an experiment config and runs a small
training loop with dummy data under XLA compilation. The StableHLO text for
that compiled training step sequence is written to the specified output file.

Example:
```
python -m official.nlp.tools.dump_stablehlo \
    --experiment=bert/sentence_prediction \
    --output=output/bert_train.stablehlo \
    --num_devices=2 \
    --iterations=2
```
"""
from absl import app
from absl import flags
import os
import sys
import types
import tensorflow as tf, tf_keras

# Allow running the script directly via its file path.
if __package__ in (None, ''):
  sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Provide a stub tensorflow_text module if the real one isn't installed so that
# optional NLP tasks can be imported for registration.
try:
  import tensorflow_text  # pylint: disable=unused-import
except ModuleNotFoundError:
  sys.modules['tensorflow_text'] = types.ModuleType('tensorflow_text')

from official.core import exp_factory
from official.core import task_factory
from official.modeling import optimization

# Import registry modules but allow missing optional dependencies such as
# tensorflow_text. This mirrors the behavior of ``exp_factory`` when used in
# environments without all NLP extras installed.
try:  # pylint: disable=unused-import
  from official.common import registry_imports  # noqa: F401
except ModuleNotFoundError as exc:  # tensorflow_text may be absent
  if 'tensorflow_text' in str(exc):
    sys.stderr.write('WARNING: tensorflow_text not available; ' +
                     'translation tasks will not be registered.\n')
  else:
    raise

FLAGS = flags.FLAGS

flags.DEFINE_string('experiment', None, 'Experiment config name.')
flags.DEFINE_string('output', 'stablehlo.txt', 'File to write StableHLO text.')
flags.DEFINE_integer('batch_size', 16, 'Global batch size for dummy inputs.')
flags.DEFINE_integer('num_devices', 2, 'Number of logical devices for MirroredStrategy.')
flags.DEFINE_integer('iterations', 4,
                     'Number of training iterations to include in the StableHLO.')


def main(_):
  params = exp_factory.get_exp_config(FLAGS.experiment)
  params.runtime.enable_xla = True
  params.task.train_data.input_path = 'dummy'
  params.task.train_data.global_batch_size = FLAGS.batch_size
  if hasattr(params.task.model, "num_classes"):
    params.task.model.num_classes = max(1, params.task.model.num_classes)
  # Simplify optimizer configuration to avoid dependencies on the global step.
  params.trainer.optimizer_config.learning_rate.type = 'constant'
  params.trainer.optimizer_config.learning_rate.constant.learning_rate = 0.0
  params.trainer.optimizer_config.warmup.type = None

  if FLAGS.num_devices > 1:
    strategy = tf.distribute.MirroredStrategy()
  else:
    strategy = tf.distribute.get_strategy()

  with strategy.scope():
    task = task_factory.get_task(params.task)
    model = task.build_model()
    opt_factory = optimization.OptimizerFactory(params.trainer.optimizer_config)
    learning_rate = opt_factory.build_learning_rate()
    optimizer = opt_factory.build_optimizer(learning_rate)
    metrics = task.build_metrics(training=True)

    # Replace apply_gradients with a version that performs explicit
    # cross-replica summation using ReplicaContext.all_reduce. This allows
    # gradient aggregation to be captured in the StableHLO.
    _orig_apply_gradients = optimizer.apply_gradients

    def _apply_gradients_all_reduce(grads_and_vars, name=None,
                                    experimental_aggregate_gradients=True):
      grads_and_vars = list(grads_and_vars)
      grads, vars = zip(*grads_and_vars)
      replica_ctx = tf.distribute.get_replica_context()
      if replica_ctx:
        reduced = []
        for g in grads:
          if g is None:
            reduced.append(None)
            continue
          if isinstance(g, tf.IndexedSlices):
            g = tf.convert_to_tensor(g)
          reduced.append(
              replica_ctx.all_reduce(tf.distribute.ReduceOp.SUM, g))
        grads_and_vars = list(zip(reduced, vars))
      return _orig_apply_gradients(
          grads_and_vars,
          name=name,
          experimental_aggregate_gradients=False)

    optimizer.apply_gradients = _apply_gradients_all_reduce

  @tf.function(jit_compile=True)
  def train_loop(inputs):
    logs = task.train_step(inputs,
                           model=model,
                           optimizer=optimizer,
                           metrics=metrics)
    for _ in tf.range(FLAGS.iterations - 1):
      logs = task.train_step(inputs,
                             model=model,
                             optimizer=optimizer,
                             metrics=metrics)
    return logs

  def distributed_step(inputs):
    return strategy.run(train_loop, args=(inputs,))

  dummy_ds = task.build_inputs(params.task.train_data)
  distributed_ds = strategy.experimental_distribute_dataset(dummy_ds)
  sample = next(iter(distributed_ds))

  # distributed_step(sample)

  def _first_replica_value(x):
    if isinstance(x, tf.distribute.DistributedValues):
      return strategy.experimental_local_results(x)[0]
    return x

  # Get the StableHLO for the compiled training loop while still in replica context
  def _get_ir_fn(inputs):
    return train_loop.experimental_get_compiler_ir(inputs)

  ir_fn = strategy.run(_get_ir_fn, args=(sample,))
  ir_fn = strategy.experimental_local_results(ir_fn)[0]

  local_sample = tf.nest.map_structure(_first_replica_value, sample)
  hlo_text = ir_fn(stage='stablehlo')

  with tf.io.gfile.GFile(FLAGS.output, 'w') as f:
    f.write(hlo_text)
  print('StableHLO written to', FLAGS.output)


if __name__ == '__main__':
  flags.mark_flag_as_required('experiment')
  app.run(main)
