variants = []
for variant in sweeper.iterate_hyperparameters():
    variants.append(variant)

def process_args(variant):
    variant['env_kwargs']['config']['seedid'] = variant['seedid']
    if not args.use_ray:
        variant['render'] = args.no_render
        if args.job in ['exp']:
            variant['algorithm_args']['num_eval_steps_per_epoch'] = 0
            variant['algorithm_args']['dump_tabular'] = args.no_dump_tabular
        elif args.job in ['demo']:
            variant['demo_kwargs']['num_episodes'] = 10
        elif args.job in ['practice']:
            variant['demo_kwargs']['num_episodes'] = 10
    if args.job in ['demo']:
        variant['env_kwargs']['config']['adapts'] = []

if args.use_ray:
    import ray
    from ray.util import ActorPool
    from itertools import cycle,count
    ray.init(temp_dir='/tmp/ray_exp', num_gpus=args.gpus)

    @ray.remote
    class Iterators:
        def __init__(self):
            self.run_id_counter = count(0)
        def next(self):
            return next(self.run_id_counter)
    iterator = Iterators.options(name="global_iterator").remote()

    @ray.remote(num_cpus=1,num_gpus=1/args.per_gpu if args.gpus else 0)
    class Runner:
        def run(self,variant):
            ptu.set_gpu_mode(True)
            iterator = ray.get_actor("global_iterator")
            run_id = ray.get(iterator.next.remote())
            variant['launcher_config']['gpu_id'] = 0
            run_variants(experiment, [variant], process_args,run_id=run_id)
    runners = [Runner.remote() for i in range(args.gpus*args.per_gpu)]
    runner_pool = ActorPool(runners)
    list(runner_pool.map(lambda a,v: a.run.remote(v), variants))
else:
    import time
    current_time = time.time_ns()
    variant = variants[0]
    run_variants(experiment, [variant], process_args,run_id=str(current_time))