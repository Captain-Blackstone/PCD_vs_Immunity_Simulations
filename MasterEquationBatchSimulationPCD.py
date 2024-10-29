from MasterEquationSimulationPCD import PCDSimulation
from command_line_interface_functions import *
import atexit
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="MasterEquation simulator PCD")
    tune_parser(parser)
    parser.add_argument("--nondivision_threshold", type=int, default=1)
    parser.add_argument("--phage_influx", type=float, default=0)
    parser.add_argument("--refine", type=float, default=0)
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
    if args.refine != 0:
        max_r = None
        if Path(f"{save_path}/population_size_estimate.txt").exists():
            estimates = pd.read_csv(f"{save_path}/population_size_estimate.txt", sep=",", header=None)
            max_r = estimates.loc[estimates[2] == estimates[2].max()][1].values[0]
        a_neutral = scan_grid(params={"A": args.A, "B": args.B, "C": args.C,
                                      "D": args.D, "E": args.E, "F": args.F,
                                      "G": args.death_function_curvature, "T": args.death_function_threshold},
                              r_steps=args.r, a_steps=args.a,
                              path=save_path,
                              simulationClass=PCDSimulation,
                              mode=args.mode,
                              discretization_volume=args.discretization_volume,
                              discretization_damage=args.discretization_damage,
                              nondivision_threshold=args.nondivision_threshold,
                              phage_influx=args.phage_influx,
                              a_min=args.refine,
                              a_max=1,
                              max_r=max_r)

    else:
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

        find_the_peak_pcd(params=params,
                          path=save_path,
                          mode=args.mode,
                          a_steps=args.a,
                          simulationClass=PCDSimulation,
                          discretization_volume=args.discretization_volume,
                          discretization_damage=args.discretization_damage,
                          nondivision_threshold=args.nondivision_threshold,
                          phage_influx=args.phage_influx
                          )
        get_landscape_contour(params=params,
                              path=save_path,
                              simulationClass=PCDSimulation,
                              mode=args.mode,
                              discretization_volume=args.discretization_volume,
                              discretization_damage=args.discretization_damage,
                              nondivision_threshold=args.nondivision_threshold,
                              phage_influx=args.phage_influx)
    with open(f"{save_path}/scanning.txt", "a") as fl:
        fl.write("success\n")
