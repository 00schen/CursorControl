from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
import gtimer as gt
import numpy as np
import torch
from rlkit.core import logger, eval_util
from rlkit.core.rl_algorithm import _get_epoch_timings


class TorchBatchRewRLAlgorithm(TorchBatchRLAlgorithm):
    def __init__(self, rew_trainer, num_rew_trains_per_train_loop=1, **kwargs):
        super().__init__(**kwargs)
        self.rew_trainer = rew_trainer
        self.num_rew_trains_per_train_loop = num_rew_trains_per_train_loop

    def _train(self):
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.eval_data_collector.collect_new_paths(
                self.eval_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling')

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                gt.stamp('exploration sampling', unique=False)
                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                self.rew_trainer.eval_new_paths(new_expl_paths)

                for _ in range(self.num_rew_trains_per_train_loop):
                    train_data = self.replay_buffer.random_balanced_batch(
                        self.batch_size)
                    self.rew_trainer.train(train_data)

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(self.batch_size)
                    train_data['rewards'] = np.log(np.sum(self.rew_trainer.rew_net(torch.from_numpy(
                        train_data['observations'].astype(np.float32))).detach().numpy() * train_data['actions'],
                                                          axis=1, keepdims=True))
                    train_data['rewards'] = np.log(self.rew_trainer.rew_net(torch.from_numpy(
                        train_data['next_observations'].astype(np.float32))).detach().numpy())
                    train_data['rewards'] = np.maximum(train_data['rewards'], -5)  # TODO: set min reward param
                    self.trainer.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        logger.record_dict(
            self.replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')
        logger.record_dict(self.rew_trainer.get_diagnostics(), prefix='rew_trainer/')

        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(),
            prefix='exploration/'
        )
        expl_paths = self.expl_data_collector.get_epoch_paths()
        if hasattr(self.expl_env, 'get_diagnostics'):
            logger.record_dict(
                self.expl_env.get_diagnostics(expl_paths),
                prefix='exploration/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(expl_paths),
            prefix="exploration/",
        )
        """
        Evaluation
        """
        logger.record_dict(
            self.eval_data_collector.get_diagnostics(),
            prefix='evaluation/',
        )
        # eval_paths = self.eval_data_collector.get_epoch_paths()
        # breakpoint()
        # if hasattr(self.eval_env, 'get_diagnostics'):
        #     logger.record_dict(
        #         self.eval_env.get_diagnostics(eval_paths),
        #         prefix='evaluation/',
        #     )
        # logger.record_dict(
        #     eval_util.get_generic_path_information(eval_paths),
        #     prefix="evaluation/",
        # )

        """
        Misc
        """
        gt.stamp('logging')
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    def _end_epoch(self, epoch):
        super()._end_epoch(epoch)
        self.rew_trainer.end_epoch(epoch)
