import argparse
import pandas as pd
from generate import search_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", default="./check.csv", type=str, help="input path"
    )
    parser.add_argument(
        "--output_path", default="./output.csv", type=str, help="output path"
    )
    parser.add_argument("--iters", default=50, type=int, help="iteraions")
    args = parser.parse_args()

    df = pd.read_csv(args.input_path, header=None)
    initial_pop = df[0].tolist()

    smiles_pop = initial_pop.copy()
    for i in range(len(initial_pop)):
        print(i)
        smiles_pop = search_step(smiles_pop)

    pd.DataFrame(smiles_pop, columns=['output']).to_csv(
        args.output_path, index=None, header=None)


if __name__ == "__main__":
    main()
