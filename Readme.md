# Deep Implicit Templates for 3D Shape Representation
Zerong Zheng, Tao Yu, Qionghai Dai, Yebin Liu. arXiv 2020. 

This repository is an implementation for [Deep Implicit Templates](http://www.liuyebin.com/dit/dit.html). 
Full paper is available [here](https://arxiv.org/abs/2011.14565). 

![Teaser Image](http://www.liuyebin.com/dit/assets/results1.jpg)

## Citing DIT

If you use DIT in your research, please cite the
[paper](https://arxiv.org/abs/2011.14565):
```
@misc{zheng2020dit,
title={Deep Implicit Templates for 3D Shape Representation},
author={Zheng, Zerong and Yu, Tao and Dai, Qionghai and Liu, Yebin},
year={2020},
eprint={2011.14565},
archivePrefix={arXiv},
primaryClass={cs.CV}
}
```

## Requirements
* Ubuntu 18.04 
* Pytorch (tested on 1.7.0)
* plyfile
* matplotlib
* ninja
* pathos
* tensorboardX
* pyrender
  
## Demo
This repo contains pre-trained models for cars, chairs, airplanes and sofas. 
After cloning the code repo, please run the following commands to generate the sofa template as well as 20 training sofa meshes with the color-coded canonical coordinates (i.e., the correspondences between the template and the meshes). 
```bash
GPU_ID=0
CUDA_VISIBLE_DEVICES=${GPU_ID} python generate_template_mesh.py -e pretrained/sofas_dit --debug 
CUDA_VISIBLE_DEVICES=${GPU_ID} python generate_training_meshes.py -e pretrained/sofas_dit --debug --start_id 0 --end_id 20 --octree --keep_normalization
CUDA_VISIBLE_DEVICES=${GPU_ID} python generate_meshes_correspondence.py -e pretrained/sofas_dit --debug --start_id 0 --end_id 20
```
The canonical coordinates are stored as float RGB values in ```.ply``` files. You can render the color-coded meshes for visualization by running: 
```bash
python render_correspondences.py  -i pretrained/sofas_dit/TrainingMeshes/2000/ShapeNet/[....].ply
```

## Data Preparation

Please follow original setting of [DeepSDF](https://github.com/facebookresearch/DeepSDF) to prepare the SDF data in ```./data``` folder.


## Traing and Evaluation

After preparing the data following DeepSDF, you can train the model as:
```bash
GPU_ID=0
preprocessed_data_dir=./data
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_deep_implicit_templates.py -e examples/sofas_dit --debug --batch_split 2 -c latest -d ${preprocessed_data_dir}
```

To evaluate the reconstruction accuracy (Tab.2 in our paper), please run: 
```bash
GPU_ID=0
preprocessed_data_dir=./data
CUDA_VISIBLE_DEVICES=${GPU_ID} python reconstruct_deep_implicit_templates.py -e examples/sofas_dit -c 2000 --split examples/splits/sv2_sofas_test.json -d ${preprocessed_data_dir} --skip --octree
CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluate.py -e examples/sofas_dit -c 2000 -s examples/splits/sv2_sofas_test.json -d ${preprocessed_data_dir} --debug
```

Due the the randomness of the points sampled from the meshes, the numeric results will vary across multiple reruns of the same shape, 
and will likely differ from those produced in the paper. 


## Acknowledgements
This code repo is heavily based on [DeepSDF](https://github.com/facebookresearch/DeepSDF). We thank the authors for their great job!



## License
MIT License
