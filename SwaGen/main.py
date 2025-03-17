from train import main as train_main
from simulation import run_simulations
import numpy as np


def main():
    # Run the training workflow
    model = train_main()

    # Optionally, run the simulation (if needed)
    res_array_naive = []
    res_array_predicted = []
    run_simulations(res_array_naive, res_array_predicted, model)
    print("Simulation Results:")
    print("Naive:", res_array_naive)
    print("Predicted:", res_array_predicted)
    print("Mean:", np.mean(res_array_naive))
    print("Variance:", np.var(res_array_naive))


if __name__ == '__main__':
    main()
