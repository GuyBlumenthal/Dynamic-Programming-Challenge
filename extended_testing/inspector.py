import pickle
import numpy as np

def inspect_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print(f"Contents of {file_path}:")
            print(data)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

def inspect_npz_file(file_path):
    try:
        with np.load(file_path, allow_pickle=True) as data:
            print(f"Contents of {file_path}:")
            for key in data.files:
                print(f"  {key}: {data[key]}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    inspect_pickle_file('../tests/test0.pkl')
    inspect_npz_file('../tests/test0.npz')