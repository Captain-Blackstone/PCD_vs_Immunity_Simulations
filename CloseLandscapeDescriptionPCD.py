from MasterEquationSimulationPCD import PCDSimulation
from command_line_interface_functions import *
import atexit
import logging
import argparse
from pathlib import Path
def scan_grid_log(params: dict,
                  r_steps: int,
                  a_steps: int,
                  path: str,
                  simulationClass,
                  aa=None,
                  rr=None,
                  **kwargs):
    parameters = params.copy()
    estimates_file = f"{path}/population_size_estimate.txt"
    if aa is None:
        aa = np.linspace(0, 1, a_steps)
    if rr is None:
        # rr = [0] + list(np.logspace(0, 2, r_steps-1) / 100 * params["E"])
        rr = np.linspace(0, params["E"], r_steps)
    for r in rr:
        conditions = initialize_conditions_dictionary(simulationClass)
        for a in aa:
            # Do not rerun already existing estimations
            current_estimate = get_estimate(file=estimates_file, a_val=a, r_val=r)
            if current_estimate is not None:
                continue

            parameters["a"] = round(a, 10)
            parameters["r"] = round(r, 10)

            if (any([el is None for el in conditions.values()]) or
                    (conditions["matrix"] is not None and conditions["matrix"].sum() < 1)):
                simulation = simulationClass(params=parameters, save_path=path, **kwargs)
                simulation.run(1000000000000000000, save=False)
                for key in conditions.keys():
                    conditions[key] = getattr(simulation, key)

            if all([el is not None for el in conditions.values()]) and conditions["matrix"].sum() > 0:
                # Don't start from dead population
                if conditions["matrix"].sum() < 1:
                    conditions["matrix"] = conditions["matrix"] / conditions["matrix"].sum()
                # If no phage in chemostat, add some phage
                if conditions.get("ksi") is not None and conditions["ksi"] < 1:
                    conditions["ksi"] = 10000

            logging.info(f"starting simulation with params: {parameters}")
            # Create simulation
            simulation = simulationClass(params=parameters, save_path=path, **kwargs)

            # Initialize with conditions
            for key, val in conditions.items():
                setattr(simulation, key, val)

            # Run simulation
            simulation.run(1000000000000000000)

            # Save the final conditions
            for key in conditions.keys():
                conditions[key] = getattr(simulation, key)


logging.basicConfig(level=logging.WARNING)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="MasterEquation simulator PCD")
    tune_parser(parser)
    parser.add_argument("--nondivision_threshold", type=int, default=1)
    parser.add_argument("--phage_influx", type=float, default=0)
    parser.add_argument("--refine", type=int, default=0)
    parser.add_argument("-dft", "--death_function_threshold", type=float, default=1)
    parser.add_argument("-dfc", "--death_function_curvature", type=float, default=1)
    args = parser.parse_args()
    if args.debug:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO)

    if args.mode in ["local", "interactive"]:
        save_path = f"../../data/master_equation/" \
                    f"{args.A}_{args.B}_{args.C}_{args.D}_{args.phage_influx}_{args.E}_{args.F}_{args.death_function_threshold}_{args.death_function_curvature}"
    else:
        save_path = f"./data/{args.A}_{args.B}_{args.C}_{args.D}_{args.phage_influx}_{args.E}_{args.F}_{args.death_function_threshold}_{args.death_function_curvature}"
    Path(save_path).mkdir(exist_ok=True)
    atexit.register(lambda: write_completion(save_path))
    if get_estimate(file=f"{save_path}/population_size_estimate.txt", a_val=0, r_val=0) is None:
        PCDSimulation(mode=args.mode,
                          params={"A": args.A, "B": args.B, "C": args.C, "D": args.D,
                                   "E": args.E, "F": args.F,
                                   "G": args.death_function_curvature, "T": args.death_function_threshold,
                                   "a": 0, "r": 0},
                           save_path=save_path,
                           discretization_volume=args.discretization_volume,
                           discretization_damage=args.discretization_damage,
                           nondivision_threshold=args.nondivision_threshold,
                           phage_influx=args.phage_influx
                           ).run(100000000, save=True)
    params = {"A": args.A, "B": args.B, "C": args.C,
                  "D": args.D, "E": args.E, "F": args.F,
                  "G": args.death_function_curvature, "T": args.death_function_threshold}
    scan_grid_log(params=params,
            r_steps=args.r, a_steps=args.a,
                      path=save_path,
                      simulationClass=PCDSimulation,
                      mode=args.mode,
                      discretization_volume=args.discretization_volume,
                      discretization_damage=args.discretization_damage,
                      nondivision_threshold=args.nondivision_threshold,
                      phage_influx=args.phage_influx,
                      )

    with open(f"{save_path}/scanning.txt", "a") as fl:
        fl.write("success\n")
