#### To run the training and Fisher information computation, use the following command:
python scripts/train_and_compute_fisher.py --save_dir saved_models --model_name my_model

#### To compute the entropy with a specified temperature, use the command:
python scripts/compute_entropy.py --model_path saved_models/my_model --temperature 0.05