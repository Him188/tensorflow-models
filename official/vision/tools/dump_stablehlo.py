"""Utility to dump StableHLO for a vision model training step.

This script builds a model from an experiment config and runs a single
training step with dummy data under XLA compilation. The StableHLO text
for the compiled training step is written to the specified output file.

Example:
```
python -m official.vision.tools.dump_stablehlo \
    --experiment=mobilenet_imagenet \
    --output=/tmp/mobilenet_train.stablehlo
```
"""
from absl import app
from absl import flags
import tensorflow as tf, tf_keras

from official.core import exp_factory
from official.core import task_factory
from official.modeling import optimization
from official.vision import registry_imports  # pylint: disable=unused-import
from official.vision.ops import anchor
from official.vision.configs import image_classification as ic_cfg
from official.vision.configs import maskrcnn as maskrcnn_cfg
from official.vision.configs import retinanet as retinanet_cfg
from official.vision.configs import semantic_segmentation as seg_cfg

FLAGS = flags.FLAGS

flags.DEFINE_string('experiment', None, 'Experiment config name.')
flags.DEFINE_string('output', 'stablehlo.txt', 'File to write StableHLO text.')
flags.DEFINE_integer('batch_size', 1, 'Batch size for dummy inputs.')


def _dummy_maskrcnn_labels(model_cfg: maskrcnn_cfg.MaskRCNN,
                           data_cfg: maskrcnn_cfg.DataConfig) -> dict:
  """Creates dummy labels for MaskRCNN models."""
  max_instances = data_cfg.parser.max_num_instances
  mask_crop = data_cfg.parser.mask_crop_size
  anchors = anchor.build_anchor_generator(
      model_cfg.min_level,
      model_cfg.max_level,
      model_cfg.anchor.num_scales,
      model_cfg.anchor.aspect_ratios,
      model_cfg.anchor.anchor_size,
  )(model_cfg.input_size[:2])
  anchors_per_loc = model_cfg.anchor.num_scales * len(model_cfg.anchor.aspect_ratios)
  rpn_score_targets = {}
  rpn_box_targets = {}
  for level, boxes in anchors.items():
    h, w, _ = boxes.shape
    rpn_score_targets[level] = tf.zeros([
        FLAGS.batch_size, h, w, anchors_per_loc
    ], tf.float32)
    rpn_box_targets[level] = tf.zeros([
        FLAGS.batch_size, h, w, anchors_per_loc * 4
    ], tf.float32)
  labels = {
      'image_info': tf.zeros([FLAGS.batch_size, 4, 2], tf.float32),
      'anchor_boxes': anchors,
      'rpn_score_targets': rpn_score_targets,
      'rpn_box_targets': rpn_box_targets,
      'gt_boxes': tf.zeros([FLAGS.batch_size, max_instances, 4], tf.float32),
      'gt_classes': tf.zeros([FLAGS.batch_size, max_instances], tf.int32),
  }
  if model_cfg.include_mask:
    labels['gt_masks'] = tf.zeros(
        [FLAGS.batch_size, max_instances, mask_crop, mask_crop], tf.float32)
    if model_cfg.outer_boxes_scale > 1.0:
      labels['gt_outer_boxes'] = tf.zeros([
          FLAGS.batch_size, max_instances, 4
      ], tf.float32)
  return labels


def _dummy_retinanet_labels(model_cfg: retinanet_cfg.RetinaNet,
                             data_cfg: retinanet_cfg.DataConfig) -> dict:
  """Creates dummy labels for RetinaNet models."""
  anchors = anchor.build_anchor_generator(
      model_cfg.min_level,
      model_cfg.max_level,
      model_cfg.anchor.num_scales,
      model_cfg.anchor.aspect_ratios,
      model_cfg.anchor.anchor_size,
  )(model_cfg.input_size[:2])
  anchors_per_loc = (
      model_cfg.anchor.num_scales * len(model_cfg.anchor.aspect_ratios))
  cls_targets = {}
  box_targets = {}
  num_anchors = 0
  for level, boxes in anchors.items():
    h, w, _ = boxes.shape
    cls_targets[level] = tf.zeros([h, w, anchors_per_loc], tf.float32)
    box_targets[level] = tf.zeros([h, w, anchors_per_loc * 4], tf.float32)
    num_anchors += h * w * anchors_per_loc
  labels = {
      'image_info': tf.zeros([FLAGS.batch_size, 4, 2], tf.float32),
      'anchor_boxes': anchors,
      'cls_targets': cls_targets,
      'box_targets': box_targets,
      'cls_weights': tf.zeros([FLAGS.batch_size, num_anchors], tf.float32),
      'box_weights': tf.zeros([FLAGS.batch_size, num_anchors], tf.float32),
  }
  return labels


def _dummy_segmentation_labels(model_cfg: seg_cfg.SemanticSegmentationModel,
                               data_cfg: seg_cfg.DataConfig) -> dict:
  """Creates dummy labels for segmentation models."""
  label_shape = model_cfg.input_size[:2]
  return {'masks': tf.zeros([FLAGS.batch_size] + label_shape + [1], tf.int32)}


def main(_):
  params = exp_factory.get_exp_config(FLAGS.experiment)
  params.task.train_data.global_batch_size = FLAGS.batch_size

  input_size = list(params.task.model.input_size)
  num_classes = getattr(params.task.model, 'num_classes', 1)

  strategy = tf.distribute.get_strategy()
  with strategy.scope():
    task = task_factory.get_task(params.task)
    model = task.build_model()
    opt_factory = optimization.OptimizerFactory(params.trainer.optimizer_config)
    learning_rate = opt_factory.build_learning_rate()
    optimizer = opt_factory.build_optimizer(learning_rate)
    metrics = task.build_metrics(training=True)

  @tf.function(jit_compile=True)
  def train_step(features, labels):
    return task.train_step((features, labels),
                           model=model,
                           optimizer=optimizer,
                           metrics=metrics)

  dummy_images = tf.random.uniform([FLAGS.batch_size] + input_size, dtype=tf.float32)
  if isinstance(params.task.model, maskrcnn_cfg.MaskRCNN):
    dummy_labels = _dummy_maskrcnn_labels(params.task.model, params.task.train_data)
  elif isinstance(params.task.model, retinanet_cfg.RetinaNet):
    dummy_labels = _dummy_retinanet_labels(params.task.model, params.task.train_data)
  elif isinstance(params.task.model, seg_cfg.SemanticSegmentationModel):
    dummy_labels = _dummy_segmentation_labels(params.task.model, params.task.train_data)
  else:
    dummy_labels = tf.random.uniform([FLAGS.batch_size],
                                     maxval=num_classes,
                                     dtype=tf.int32)

  train_step(dummy_images, dummy_labels)
  hlo_text = train_step.experimental_get_compiler_ir(
      dummy_images, dummy_labels)(stage='stablehlo')

  with tf.io.gfile.GFile(FLAGS.output, 'w') as f:
    f.write(hlo_text)
  tf.print('StableHLO written to', FLAGS.output)


if __name__ == '__main__':
  flags.mark_flag_as_required('experiment')
  app.run(main)

