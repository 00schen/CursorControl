import abc

import gtimer as gt
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector

class BatchRLAlgorithm(TorchBatchRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
        self,
        *args,
        pure_expl_data_collector,
        pure_expl_num_steps,
        session_buffer: ReplayBuffer,
        **kwargs,
    ):
        super().__init__(
            *args,**kwargs
        )
        self.pure_expl_data_collector = pure_expl_data_collector
        self.pure_expl_num_steps = pure_expl_num_steps
        self.session_buffer = session_buffer

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

        # collect pure exploration
        for _ in range(self.pure_expl_num_steps//(10*self.max_path_length)):
            self.expl_env.new_goal()
            pure_expl_paths = self.pure_expl_data_collector.collect_new_paths(
                self.max_path_length,
                10*self.max_path_length,
                discard_incomplete_paths=False,
            )
            self.expl_data_collector.end_epoch(-1)
            self.replay_buffer.add_paths(pure_expl_paths)
            if self.expl_env.goal_reached:
                self.session_buffer.add_paths(pure_expl_paths)
        gt.stamp('pretrain exploring', unique=False)

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
                    new_expl_paths = self.expl_data_collector.collect_new_paths(
                        self.max_path_length,
                        self.num_expl_steps_per_train_loop,
                        discard_incomplete_paths=False,
                    )
                    gt.stamp('exploration sampling', unique=False)

                    self.replay_buffer.add_paths(new_expl_paths)
                    if self.expl_env.goal_reached:
                        self.session_buffer.add_paths(new_expl_paths)
                    gt.stamp('data storing', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    if self.session_buffer.num_steps_can_sample():
                        sup_data = self.session_buffer.random_batch(
                            self.batch_size)
                        for k in ['observations','curr_gaze_features','curr_goal']:
                            train_data['sup_'+k] = sup_data[k]
                    self.trainer.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)
