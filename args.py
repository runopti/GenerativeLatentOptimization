import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="trained_models")
parser.add_argument("--model", type=str, default="models/mlp")
parser.add_argument("--dataset_path", type=str, default="./dataset/")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_epoch", type=int, default=30)
parser.add_argument("--max_loop_z", type=int, default=3)


args = parser.parse_args()

