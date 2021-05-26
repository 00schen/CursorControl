import abc

from rl.misc.calibration_rl_algorithm import BatchRLAlgorithm as TorchCalibrationRLAlgorithm
import gtimer as gt


class BatchRLAlgorithm(TorchCalibrationRLAlgorithm, metaclass=abc.ABCMeta):
    def _train(self):
        # For some reason, pybullet crashes for me when pre-train path collect isn't used.
        # So a token sample is taken
        self.expl_env.new_goal()
        self.expl_data_collector.collect_new_paths(
            self.max_path_length,
            10,
            discard_incomplete_paths=False,
        )
        self.expl_data_collector.end_epoch(-1)

        self.expl_env.per_step = True
        self.eval_env.per_step = True

        # calibrate
        for index in self.calibration_indices:
            self.expl_env.new_goal(index)
            for _ in range(self.trajs_per_index):
                calibration_paths = self.calibration_data_collector.collect_new_paths(
                    self.max_path_length,
                    1,
                    discard_incomplete_paths=False,
                )
                self.expl_data_collector.end_epoch(-1)
                self.calibration_buffer.add_paths(calibration_paths)
        gt.stamp('pretrain exploring', unique=False)
        self._sample_and_train(self.pretrain_steps, self.calibration_buffer)

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

                    # switch positions do not change
                    failures = 0
                    while True:
                        new_expl_paths = self.expl_data_collector.collect_new_paths(
                            self.max_path_length,
                            1,
                            discard_incomplete_paths=False,
                        )

                        gt.stamp('exploration sampling', unique=False)

                        # should be only 1 path
                        assert len(new_expl_paths) == 1

                        self.replay_buffer.add_paths(new_expl_paths)
                        gt.stamp('data storing', unique=False)

                        if self.replay_buffer.num_steps_can_sample():
                            # abuse of naming, num_trains_per_train_loop is actually # trains per new episode
                            self._sample_and_train(self.num_trains_per_train_loop, self.replay_buffer)

                        if new_expl_paths[0]['env_infos'][-1]['task_success']:
                            break
                        else:
                            failures += 1
                            if failures >= self.max_failures:
                                break

            self._end_epoch(epoch)
