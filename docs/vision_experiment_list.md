# Vision Experiments

The following experiments are registered in the Vision library and can be used with `dump_stablehlo.py`:

- `cascadercnn_spinenet_coco`
- `deit_imagenet_pretrain`
- `fasterrcnn_resnetfpn_coco`
- `image_classification`
- `maskrcnn_mobilenet_coco`
- `maskrcnn_resnetfpn_coco`
- `maskrcnn_spinenet_coco`
- `mnv2_deeplabv3_cityscapes`
- `mnv2_deeplabv3_pascal`
- `mnv2_deeplabv3plus_cityscapes`
- `mobilenet_imagenet`
- `resnet_imagenet`
- `resnet_rs_imagenet`
- `retinanet`
- `retinanet_mobile_coco`
- `retinanet_resnetfpn_coco`
- `retinanet_spinenet_coco`
- `revnet_imagenet`
- `seg_deeplabv3_pascal`
- `seg_deeplabv3plus_cityscapes`
- `seg_deeplabv3plus_pascal`
- `seg_resnetfpn_pascal`
- `semantic_segmentation`
- `video_classification`
- `video_classification_kinetics400`
- `video_classification_kinetics600`
- `video_classification_kinetics700`
- `video_classification_kinetics700_2020`
- `video_classification_ucf101`
- `vit_imagenet_finetune`
- `vit_imagenet_pretrain`

Use `python -m official.vision.tools.collect_all_stablehlo` to produce StableHLO files for each of these experiments. Results for each batch size (`1, 4, 8, 16, 64`) are written under the `output/BATCHSIZE/` directory by default.
