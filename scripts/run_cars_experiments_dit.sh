# train the networks (also generate the training mesh for debugging)
GPU_ID=0
preprocessed_data_dir=../ShapeNet/DeepSDF
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_deep_implicit_templates.py -e examples/cars_dit --debug --batch_split 2 -c latest -d ${preprocessed_data_dir}
CUDA_VISIBLE_DEVICES=${GPU_ID} python reconstruct_deep_implicit_templates.py -e examples/cars_dit -c 2000 --split examples/splits/sv2_cars_test.json -d ${preprocessed_data_dir} --skip --octree
CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluate.py -e examples/cars_dit -c 2000 -s examples/splits/sv2_cars_test.json -d ${preprocessed_data_dir} --debug

#CUDA_VISIBLE_DEVICES=${GPU_ID} python train_deep_implicit_templates.py -e examples/cars_dit_no_curriculum --debug -d ${preprocessed_data_dir}
#CUDA_VISIBLE_DEVICES=${GPU_ID} python reconstruct_deep_implicit_templates.py -e examples/cars_dit_no_curriculum -c 2000 --split examples/splits/sv2_cars_test.json -d ${preprocessed_data_dir} --skip --octree
#CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluate.py -e examples/cars_dit_no_curriculum -c 2000 -d ${preprocessed_data_dir} -s examples/splits/sv2_cars_test.json
