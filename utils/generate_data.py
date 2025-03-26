import numpy as np
import argparse
import os
from utils.data_utils import save_dataset
from TspInstanceFileTool import TspInstanceFileTool

def generate_mtsp_data(dataset_size, tsp_size):
    return np.random.uniform(size=(dataset_size, tsp_size, 2)).tolist()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default="mtsplib", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='../graph', help="Create datasets in data_dir/problem (default 'data')")

    parser.add_argument("--dataset_size", type=int, default=100, help="Size of the dataset")
    parser.add_argument('--graph_size', type=int, default=50,
                        help="Sizes of problem instances (default 50)")
    parser.add_argument('--seed', type=int, default=528, help="Random seed")

    args = parser.parse_args()

    data = generate_mtsp_data(args.dataset_size, args.graph_size)

    datadir = os.path.join(args.data_dir, 'mtsp')
    os.makedirs(datadir, exist_ok=True)
    if args.filename is None:
        filename = os.path.join(datadir, "mtsp{}_{}_seed{}.pkl".format(args.graph_size, args.dataset_size, args.seed))
    else:
        filename = os.path.join(datadir, args.filename)

    if args.filename == 'mtsplib':
        dataset = []
        for graph_name in ("eil51", "berlin52", "eil76", "rat99"):
            graph, scale = TspInstanceFileTool.loadTSPLib("../graph/tsp", graph_name)
            dataset.append(graph.squeeze(0).tolist())
    else:
        np.random.seed(args.seed)
        dataset = generate_mtsp_data(args.dataset_size, args.graph_size)

    save_dataset(dataset, filename)