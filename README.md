A custom rlkit version is used that simply removes unused calls to mujoco and related libraries.
the assistive-gym branch is "SharedAutonomy"
task scripts are stored in image/rl/scripts. In the current set up, most hyperparameters are stored as variables in the main function, and shell arguments are for compute scaling.
To collect demos, run `python image/rl/scripts/collect_demo.py --env_name [environment] [--use_ray]`
    demos will show up in the demos folder in image
To run an experiment run `python image/rl/scripts/cql_experiment.py --env_name [environment] [--use_ray] --gpus [num gpus] --per_gpu [num sessions per gpu]`
    results will show up in the logs folder in image
Current environments that can be run are `LightSwitch` and `Laptop`, but adding to the environments is pretty standard python.
The environment will render is ray is not used and `--no_render` is not set in shell arguments

To visualize the environment/debug, you can use demo
