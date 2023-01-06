import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str,  help="Path to a file with labels or batch effects")
args = parser.parse_args()

with open(args.file, 'rb') as f:
    data = pickle.load(f)

for i in data:
    x,y,z =i
    print(f'{args.file}\t{x}\t{y}\t{z}')
