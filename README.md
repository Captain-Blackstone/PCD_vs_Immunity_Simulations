# PCD vs Immunity Simulations
This is a reposetory containing scripts implementing simulations used in the paper (paper name).
There are 2 main scripts the user is encouraged to interface with: <br>

<b> MasterEquationSimulationPCD.py </b>. Contains a command line interface for running a single simulation in a given environment. The values of a (investment in PCD) and r (investment in immunity) have to be specified. Use python3 MasterEquationSimulationPCD.py --help to learn about the parameters necessary for running the script. Note that --interactive flag allows to run a simulation in an interactive regime monitoring population size, structure, number of phages and nutrient availability. Using interactive mode it is also possible to change the parameters of the simulation on the run, but the simulation has to be paused while the parameters are being updated. <br>

<b> MasterEquationBatchSimulationPCD.py </b>. Contains a command line interface for scanning a whole fitness landcape for a given parameter combination. The script scans through a grid of possible values of a and r and attempts to locate the peak in the fitness landscape. Use python3 MasterEquationBatchSimulationPCD.py --help to learn about the parameters necessary for running the script. <br>

