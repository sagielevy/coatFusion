Sometimes samples recieve bad mask coverage. To validate that mask presence is above some threshold (configurable), and to verify that material properties per result exist in json file, run:
(num-coatings should match the number of coatings per sample that was configured in DatasetConfig)

python coating_dataset_generator/validate_coating_mask_presence.py coating_dataset_Benchmark --num-coatings 67 --check-jsons
python coating_dataset_generator/validate_coating_mask_presence.py coating_dataset_Training --num-coatings 64 --check-jsons


To generate specific samples by name or index:

python -m coating_dataset_generator.dataset_generation_batch_manager 0 1 --indices_list 1 150,1089,1110,1139,136,142
python -m coating_dataset_generator.dataset_generation_batch_manager 0 1 --names_list "Lion Head,Owl,Cat Head"
python -m coating_dataset_generator.dataset_generation_batch_manager 0 1 --names_list "Concrete Road Barrier"