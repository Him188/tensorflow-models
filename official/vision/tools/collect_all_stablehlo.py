import os
import subprocess
import sys

from absl import app
from absl import flags

from official.common import registry_imports  # pylint: disable=unused-import
from official.core import exp_factory

FLAGS = flags.FLAGS

flags.DEFINE_multi_integer(
    'batch_sizes', [4],
    'Batch sizes to generate StableHLO for.')
flags.DEFINE_string('output_dir', 'output',
                    'Base directory to write StableHLO text files.')


VISION_PREFIXES = (
    'image_classification',
    'resnet_imagenet',
    'resnet_rs_imagenet',
    'revnet_imagenet',
    'mobilenet_imagenet',
    'deit_imagenet_pretrain',
    'vit_imagenet_pretrain',
    'vit_imagenet_finetune',
    'fasterrcnn_resnetfpn_coco',
    'maskrcnn_resnetfpn_coco',
    'maskrcnn_spinenet_coco',
    'cascadercnn_spinenet_coco',
    'maskrcnn_mobilenet_coco',
    'retinanet',
    'retinanet_resnetfpn_coco',
    'retinanet_spinenet_coco',
    'retinanet_mobile_coco',
    'semantic_segmentation',
    'seg_deeplabv3_pascal',
    'seg_deeplabv3plus_pascal',
    'seg_resnetfpn_pascal',
    'mnv2_deeplabv3_pascal',
    'seg_deeplabv3plus_cityscapes',
    'mnv2_deeplabv3_cityscapes',
    'mnv2_deeplabv3plus_cityscapes',
    'video_classification',
    'video_classification_ucf101',
    'video_classification_kinetics400',
    'video_classification_kinetics600',
    'video_classification_kinetics700',
    'video_classification_kinetics700_2020',
)


def get_vision_experiments():
    """Returns a list of experiment names registered for vision."""
    names = []
    for k, v in exp_factory._REGISTERED_CONFIGS.items():
        if isinstance(v, dict):
            for k2 in v:
                names.append(f"{k}/{k2}")
        else:
            names.append(k)
    return sorted(n for n in names if n.startswith(VISION_PREFIXES))


def main(_):
    base_dir = FLAGS.output_dir
    os.makedirs(base_dir, exist_ok=True)
    experiments = get_vision_experiments()

    for batch in FLAGS.batch_sizes:
        config_dir = os.path.join(base_dir, str(batch))
        os.makedirs(config_dir, exist_ok=True)
        for exp in experiments:
            outfile = os.path.join(config_dir, exp.replace('/', '_') + '.stablehlo')
            if os.path.exists(outfile):
                print(f'Skipping {outfile}, already exists')
                continue
            cmd = [
                sys.executable,
                '-m',
                'official.vision.tools.dump_stablehlo',
                f'--experiment={exp}',
                f'--output={outfile}',
                f'--batch_size={batch}',
            ]
            print('Running:', ' '.join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == '__main__':
    app.run(main)
