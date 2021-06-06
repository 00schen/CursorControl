import abc

import gtimer as gt
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer


class BatchRLAlgorithm(TorchBatchRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
        self,
        *args,
        calibration_data_collector,
        calibration_indices,
        trajs_per_index,
        calibration_buffer: ReplayBuffer,
        pretrain_steps=100,
        max_failures=10,
        calibrate_split=True,
        **kwargs,
    ):
        super().__init__(
            *args,**kwargs
        )
        self.calibration_data_collector = calibration_data_collector
        if calibration_indices is None:
            calibration_indices = self.expl_env.base_env.target_indices
        self.calibration_indices = calibration_indices
        self.trajs_per_index = trajs_per_index
        self.calibration_buffer = calibration_buffer
        self.pretrain_steps = pretrain_steps
        self.max_failures = max_failures
        self.calibrate_split = calibrate_split

    def _sample_and_train(self, steps, buffer):
        self.training_mode(True)
        for _ in range(steps):
            train_data = buffer.random_batch(self.batch_size)
            self.trainer.train(train_data)
        gt.stamp('training', unique=False)
        self.training_mode(False)

    def _train(self):
        # For some reason, pybullet crashes for me when pre-train path collect isn't used.
        # So a token sample is taken
        # self.expl_env.new_goal()
        # self.expl_data_collector.collect_new_paths(
        #     self.max_path_length,
        #     10,
        #     discard_incomplete_paths=False,
        # )
        # self.expl_data_collector.end_epoch(-1)

        # calibrate
        self.expl_env.base_env.calibrate_mode(True, self.calibrate_split)

        for _ in range(self.trajs_per_index):
            for index in self.calibration_indices:
                self.expl_env.new_goal(index)
                calibration_paths = self.calibration_data_collector.collect_new_paths(
                    self.max_path_length,
                    1,
                    discard_incomplete_paths=False,
                )
                self.expl_data_collector.end_epoch(-1)
                self.calibration_buffer.add_paths(calibration_paths)
        gt.stamp('pretrain exploring', unique=False)
        self._sample_and_train(self.pretrain_steps, self.calibration_buffer)

        self.expl_env.base_env.calibrate_mode(False, False)
        self.eval_env.base_env.calibrate_mode(False, False)
        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            if self.eval_paths:
                self.eval_data_collector.collect_new_paths(
                    self.eval_path_length,
                    self.num_eval_steps_per_epoch,
                    discard_incomplete_paths=True,
                )
                gt.stamp('evaluation sampling')

            for _ in range(self.num_train_loops_per_epoch):
                if self.collect_new_paths:
                    self.expl_env.new_goal()
                    success = False
                    failed_paths = []
                    successful_paths = []

                    # switch positions do not change
                    while not success:
                        new_expl_paths = self.expl_data_collector.collect_new_paths(
                            self.max_path_length,
                            self.num_expl_steps_per_train_loop,
                            discard_incomplete_paths=False,
                        )
                        gt.stamp('exploration sampling', unique=False)
                        for path in new_expl_paths:
                            if path['env_infos'][-1]['task_success']:
                                successful_paths.append(path)
                                success = True
                            else:
                                failed_paths.append(path)
                        if len(failed_paths) >= self.max_failures:
                            break

                    # no actual relabeling right now, since goals for all paths should be same
                    # only add to paths to buffer if successful
                    if len(successful_paths):
                        self.replay_buffer.add_paths(successful_paths + failed_paths)
                        gt.stamp('data storing', unique=False)

                self._sample_and_train(self.num_trains_per_train_loop, self.replay_buffer)

            self._end_epoch(epoch)
