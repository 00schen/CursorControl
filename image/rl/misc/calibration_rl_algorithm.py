import abc

import gtimer as gt
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
import pybullet as p
from rlkit.core import logger
import rlkit.torch.pytorch_util as ptu
from pylab import *
import matplotlib

matplotlib.use("Qt5Agg")
from pygame import gfxdraw
from pathlib import Path
import cv2
import pygame
import os
import numpy as np
import torch
import math
from sklearn.svm import LinearSVR
from rl.misc.env_wrapper import real_gaze

main_dir = str(Path(__file__).resolve().parents[2])


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
            real_user=True,
            relabel_failures=True,
            seedid=0,
            curriculum=False,
            **kwargs,
    ):
        super().__init__(
            num_expl_steps_per_train_loop=1, *args, **kwargs
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
        self.real_user = real_user
        self.relabel_failures = relabel_failures
        self.seedid = seedid
        self.curriculum = curriculum
        self.blocks = []

        self.metrics = {'success_episodes': [],
                        'episode_lengths': [],
                        'success_blocks': [],
                        'block_lengths': []
                        }

        if self.real_user:
            self.metrics['correct_rewards'] = []
            self.metrics['correct_blocks'] = []
            self.metrics['user_feedback'] = []

        # self.gaze_user = GazeUser(self.expl_env)
        # self.gaze_user.run()

        if self.expl_env.env_name == 'Valve':
            self.metrics['final_angle_error'] = []
            self.metrics['init_angle_error'] = []

    def _sample_and_train(self, steps, buffers):
        self.training_mode(True)
        for _ in range(steps):
            for _ in range(len(self.trainer.vaes)):
                batches = [buffer.random_batch(self.batch_size // len(buffers)) for buffer in buffers]
                train_data = {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0].keys()}
                self.trainer.train(train_data)
        gt.stamp('training', unique=False)
        self.training_mode(False)

    def _train(self):
        # calibrate
        self.expl_env.seed(self.seedid)
        self.expl_env.base_env.calibrate_mode(True, self.calibrate_split)
        calibration_data = []

        # the buffer actually used for calibration
        calibration_buffer = self.calibration_buffer if self.calibration_buffer is not None else self.replay_buffer

        for _ in range(self.trajs_per_index):
            for index in self.calibration_indices:
                self.expl_env.new_goal(index)
                calibration_paths = self.calibration_data_collector.collect_new_paths(
                    self.max_path_length,
                    1,
                    discard_incomplete_paths=False,
                )
                self.calibration_data_collector.end_epoch(-1)

                calibration_buffer.add_paths(calibration_paths)
                calibration_data.extend(calibration_paths)

        logger.save_extra_data(calibration_data, 'calibration_data.pkl', mode='pickle')

        gt.stamp('pretrain exploring', unique=False)
        self._sample_and_train(self.pretrain_steps, [calibration_buffer])

        self.expl_env.seed(self.seedid + 100)
        self.eval_env.seed(self.seedid + 200)

        self.expl_env.base_env.calibrate_mode(False, False)
        self.eval_env.base_env.calibrate_mode(False, False)

        failed_paths = []
        successful_paths = []
        self.expl_env.new_goal()

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

            new_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_expl_steps_per_train_loop,
                discard_incomplete_paths=False,
            )
            assert len(new_expl_paths) == 1
            path = new_expl_paths[0]

            real_success = path['env_infos'][-1]['task_success']
            timeout = len(path['observations']) == self.max_path_length and not real_success

            # valve is still successful if timeout in success state
            # if self.expl_env.env_name == 'Valve' and timeout:
            #     real_success = path['env_infos'][-1]['is_success']

            gt.stamp('exploration sampling', unique=False)
            if self.real_user:
                success = None
                # automate reward if timeout and not valve env
                if timeout and not self.expl_env.env_name == 'Valve':
                    time.sleep(1)
                    success = real_success
                    self.metrics['correct_rewards'].append(None)

                elif self.expl_env.env_name == 'Valve':
                    success = path['env_infos'][-1]['feedback']

                # valve success feedback is True if user terminated as a success, -1 otherwise
                if not isinstance(success, bool):
                    while True:
                        keys = p.getKeyboardEvents()

                        if p.B3G_RETURN in keys and keys[p.B3G_RETURN] & p.KEY_WAS_TRIGGERED:
                            success = True

                            # relabel with wrong goals that were reached if not actual success.
                            # currently specific only to light switch and bottle, handled by valve differently
                            if not real_success:
                                if 'current_string' in path['env_infos'][-1]:
                                    wrong_reached_index = np.where(path['env_infos'][-1]['current_string'] == 0)[0][0]
                                    wrong_reached_goal = path['env_infos'][0]['switch_pos'][wrong_reached_index]

                                # assumes only 2 targets in bottle, and version of bottle with only 1 goal
                                elif 'unique_targets' in path['env_infos'][-1]:
                                    wrong_reached_index = 1 - path['env_infos'][-1]['unique_index']
                                    wrong_reached_goal = path['env_infos'][0]['unique_targets'][wrong_reached_index]

                                elif self.expl_env.env_name == 'Valve':
                                    wrong_reached_goal = None

                                else:
                                    raise NotImplementedError()

                                # assumes same goal each timestep
                                # not necessary when relabeling with final angle anyways
                                # if not self.expl_env.env_name == 'Valve':
                                #     for failed_path in failed_paths + [path]:
                                #         for i in range(len(failed_path['observations'])):
                                #             failed_path['observations'][i]['goal'] = wrong_reached_goal.copy()
                                #             failed_path['next_observations'][i]['goal'] = wrong_reached_goal.copy()

                            break
                        elif 8 in keys and keys[8] & p.KEY_WAS_TRIGGERED:
                            success = False
                            break

                    self.metrics['correct_rewards'].append(success == real_success)
                    self.metrics['user_feedback'].append(success)

            else:
                success = real_success

            self.metrics['success_episodes'].append(real_success)
            self.metrics['episode_lengths'].append(len(path['observations']))

            if self.expl_env.env_name == 'Valve':
                self.metrics['final_angle_error'].append(np.abs(path['env_infos'][-1]['angle_error']))
                self.metrics['init_angle_error'].append(np.abs(path['env_infos'][0]['angle_error']))

            if success:
                successful_paths.append(path)
            else:
                failed_paths.append(path)

            # no actual relabeling right now, since goals for all paths should be same
            # only add to paths to buffer if successful
            if success:
                paths_to_add = successful_paths + failed_paths if self.relabel_failures else successful_paths

                # have to relabel goals in valve with angle actually reached
                if self.expl_env.env_name == 'Valve':
                    new_target_angle = successful_paths[-1]['env_infos'][-1]['valve_angle']
                    new_goal = np.array([np.sin(new_target_angle), np.cos(new_target_angle)])
                    for path in paths_to_add:
                        for i in range(len(path['observations'])):
                            path['observations'][i]['goal'] = new_goal.copy()
                            path['next_observations'][i]['goal'] = new_goal.copy()

                self.replay_buffer.add_paths(paths_to_add)
                gt.stamp('data storing', unique=False)

                buffers = [self.replay_buffer]
                if self.calibration_buffer is not None:
                    buffers.append(self.calibration_buffer)
                self._sample_and_train(self.num_trains_per_train_loop, buffers)

            block_timeout = len(failed_paths) >= self.max_failures
            if success or block_timeout or epoch == self.num_epochs - 1:
                self.metrics['success_blocks'].append(real_success)
                self.metrics['block_lengths'].append(len(failed_paths) + len(successful_paths))
                if self.real_user:
                    self.metrics['correct_blocks'].append(block_timeout or real_success)
                self.blocks.append(failed_paths + successful_paths)

                failed_paths = []
                successful_paths = []
                self.expl_env.new_goal()  # switch positions do not change if not block end

                if self.curriculum and hasattr(self.expl_env.base_env, 'update_curriculum'):
                    self.expl_env.base_env.update_curriculum(success)

            self._end_epoch(epoch)

            logger.save_extra_data(self.metrics, 'metrics.pkl', mode='pickle')
            logger.save_extra_data(self.blocks, 'data.pkl', mode='pickle')

        log_latents = False
        if log_latents:
            total_features = []
            gaze_features = []
            policy = self.expl_data_collector._policy
            for block in [[x] for x in calibration_data] + self.blocks:
                for eps in block:
                    episode_features = []
                    episode_gaze_features = []
                    for obs in eps['observations']:
                        raw_obs = obs['raw_obs']
                        goal_set = obs.get('goal_set')
                        features = [obs[k] for k in policy.features_keys]
                        episode_gaze_features.append(np.concatenate(features))

                        if policy.incl_state:
                            features.append(raw_obs)
                            if goal_set is not None:
                                features.append(goal_set.ravel())

                        episode_features.append(np.concatenate(features))

                    last_obs = eps['next_observations'][-1]
                    raw_obs = last_obs['raw_obs']
                    goal_set = last_obs.get('goal_set')
                    features = [last_obs[k] for k in policy.features_keys]
                    episode_gaze_features.append(np.concatenate(features))

                    if policy.incl_state:
                        features.append(raw_obs)
                        if goal_set is not None:
                            features.append(goal_set.ravel())

                    episode_features.append(np.concatenate(features))

                    total_features.append(np.array(episode_features))
                    gaze_features.append(np.array(episode_gaze_features))

            for j, vae in enumerate(policy.vaes):
                latents = []
                for eps in total_features:
                    encoder_input = torch.Tensor(eps).to(ptu.device)
                    pred_features = vae.sample(encoder_input, eps=None).detach().cpu().numpy()
                    latents.append(pred_features)

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                # combined_latents = np.concatenate(latents)
                # ax.set_xlim(np.amin(combined_latents[:, 0]), np.amax(combined_latents[:, 0]))
                # ax.set_ylim(np.amin(combined_latents[:, 1]), np.amax(combined_latents[:, 1]))
                # ax.set_zlim(np.amin(combined_latents[:, 2]), np.amax(combined_latents[:, 2]))

                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])
                ax.zaxis.set_ticklabels([])

                for line in ax.xaxis.get_ticklines():
                    line.set_visible(False)
                for line in ax.yaxis.get_ticklines():
                    line.set_visible(False)
                for line in ax.zaxis.get_ticklines():
                    line.set_visible(False)

                os.makedirs(os.path.join(logger.get_snapshot_dir(), 'latents', str(j)), exist_ok=True)
                count = 1
                for eps in latents:
                    ax.set_xlim(np.amin(eps[:, 0]), np.amax(eps[:, 0]))
                    ax.set_ylim(np.amin(eps[:, 1]), np.amax(eps[:, 1]))
                    ax.set_zlim(np.amin(eps[:, 2]), np.amax(eps[:, 2]))
                    for i in range(len(eps)):
                        sc = ax.scatter(eps[:i + 1, 0], eps[:i + 1, 1], eps[:i + 1, 2],
                                        c=cm.plasma(np.arange(i + 1) / len(eps)))
                        plt.savefig(os.path.join(logger.get_snapshot_dir(), 'latents', str(j),
                                                 'latent_' + str(count) + '.png'))
                        sc.remove()
                        count += 1

            gaze_estimates = []
            for eps in gaze_features:
                estimates = np.stack((self.gaze_user.x_svr_gaze_estimator.predict(eps),
                                      self.gaze_user.y_svr_gaze_estimator.predict(eps)),
                                      axis=1)

                # convert from relative displacement from camera to normalized env coordinate system
                gaze_coords = (estimates + self.gaze_user.cam_coord[None]) * np.array([
                    [2 / self.gaze_user.window_width,
                     2 * self.gaze_user.height / (self.gaze_user.width * self.gaze_user.window_height)]])
                gaze_estimates.append(gaze_coords)


            fig = plt.figure()
            ax = fig.add_subplot(111)

            # combined_gazes = np.concatenate(gaze_estimates)
            # ax.set_xlim(np.amin(combined_gazes[:, 0]), np.amax(combined_gazes[:, 0]))
            # ax.set_ylim(np.amin(combined_gazes[:, 1]), np.amax(combined_gazes[:, 1]))

            plt.tick_params(left=False,
                            bottom=False,
                            labelleft=False,
                            labelbottom=False)

            count = 1
            for eps in gaze_estimates:
                ax.set_xlim(np.amin(eps[:, 0]), np.amax(eps[:, 0]))
                ax.set_ylim(np.amin(eps[:, 1]), np.amax(eps[:, 1]))
                for i in range(len(eps)):
                    sc = ax.scatter(eps[:i + 1, 0], eps[:i + 1, 1], c=cm.plasma(np.arange(i + 1) / len(eps)))
                    plt.savefig(os.path.join(logger.get_snapshot_dir(), 'gazes', 'gazes_' + str(count) + '.png'))
                    sc.remove()
                    count += 1

        self.expl_env.save()


class GazeUser:
    def __init__(self, env, calibration_cycles=1, cam_coord=(14.5, 0.5), window_width=29, window_height=18, **kwargs):
        self.n_samples = 10
        self.text_color = (255, 255, 255)
        self.action_text_color = (0, 0, 0)
        self.bg_color = (0, 0, 0)
        self.circle_radius = 20

        self.obs = None
        self.last_time = 0
        self.actions = None

        pygame.init()
        self.fonts = {font_size: pygame.font.Font('freesansbold.ttf', font_size) for font_size in range(2, 34, 2)}
        self.screen = pygame.display.set_mode()

        self.width, self.height = self.screen.get_size()
        self.text_field_height = self.height / 30
        self.header_coord = (self.width / 2, self.text_field_height)
        self.button_height = self.height / 20
        self.button_width = 2 * self.button_height

        for adapt in env.adapts:
            if isinstance(adapt, real_gaze):
                self.i_tracker = adapt.i_tracker
                self.face_processor = adapt.face_processor

        self.x_svr_gaze_estimator = LinearSVR(max_iter=5000)
        self.y_svr_gaze_estimator = LinearSVR(max_iter=5000)

        self.cam_coord = np.array(cam_coord)
        self.window_width = window_width
        self.window_height = window_height
        self.calibration_cycles = calibration_cycles
        self.webcam = cv2.VideoCapture(0)
        self.calibration_points = np.array([[-0.6, 0.6], [0, 0.6], [0.6, 0.6],
                                            [-0.6, 0], [0, 0], [0.6, 0],
                                            [-0.6, -0.6], [0, -0.6], [0.6, -0.6]])
        self.calibration_points[:, 0] = self.calibration_points[:, 0] * self.width / 2
        self.calibration_points[:, 1] = self.calibration_points[:, 1] * self.height / 2

    def run(self):
        signals, gaze_labels = self.record_calibration_data()
        self.calibrate(signals, gaze_labels)
        pygame.quit()

    def record_calibration_data(self):
        """
        Runs interface to collect data points for calibration.
        """
        features_list = []
        gaze_labels_list = []

        self.screen.fill(self.bg_color)
        for i, point in enumerate(self.calibration_points):
            color = (255, 255, 255)
            uncentered = self.uncenter_coord(point)
            self.draw_circle_with_text(str(i), self.action_text_color, uncentered, self.circle_radius, color)

        self.draw_rect_with_text('Press SPACE to start calibration', self.text_color, self.width,
                                 self.text_field_height, center=self.header_coord)
        pygame.display.flip()

        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        done = True
                        break

        self.draw_rect_with_text("Look at the highlighted number", self.text_color, self.width,
                                 self.text_field_height, center=self.header_coord)

        curr_point = 0
        samples_left = self.n_samples
        last_time = -1001
        cycles = 0
        started = False
        n_points = len(self.calibration_points)

        while cycles < self.calibration_cycles:
            pygame.event.get()

            curr_time = pygame.time.get_ticks()
            if not started:
                prev_point = (curr_point - 1) % n_points
                prev_coord = self.uncenter_coord(self.calibration_points[prev_point])
                self.draw_circle_with_text(str(prev_point), self.action_text_color, prev_coord,
                                           self.circle_radius, (255, 255, 255))

                curr_coord = self.uncenter_coord(self.calibration_points[curr_point])
                self.draw_circle_with_text(str(curr_point), self.action_text_color, curr_coord,
                                           self.circle_radius, (255, 165, 0))

                pygame.display.flip()
                pygame.event.get()

                started = True
                samples_left = self.n_samples
                last_time = -1001
                pygame.time.wait(1000)
            else:
                if curr_time > last_time + 100:
                    last_time = curr_time
                    _, frame = self.webcam.read()
                    features = self.face_processor.get_gaze_features(frame)

                    if features is not None:
                        features = features[:-2]
                        features_list.append(features)
                        gaze_labels_list.append(self.calibration_points[curr_point])
                        samples_left -= 1

                    if samples_left == 0:
                        curr_point += 1
                        if curr_point == len(self.calibration_points):
                            curr_point = 0
                            cycles += 1

                        started = False

        gaze_labels = np.array(gaze_labels_list)
        features = zip(*features_list)
        features = [torch.from_numpy(np.array(feature)).float().to(ptu.device) for feature in features]

        batch_size = 32
        n_batches = math.ceil(len(features_list) / batch_size)
        signals = []
        for i in range(n_batches):
            batch = [feature[i * batch_size: (i + 1) * batch_size] for feature in features]
            output = self.i_tracker(*batch)
            signals.extend(output.detach().cpu().numpy())

        signals = np.array(signals)

        return signals, gaze_labels

    def calibrate(self, signals, gaze_labels):
        """
        Fits a linear SVR gaze estimation model from the final hidden layer of
        iTracker on the collected data points.
        """
        #  convert from unnormalized env coordinate system to relative displacement from camera
        gaze_labels = gaze_labels * np.array([[self.window_width / self.width, self.window_height / self.height]])
        gaze_labels = gaze_labels - self.cam_coord[None]

        x_labels, y_labels = zip(*gaze_labels)
        self.x_svr_gaze_estimator.fit(signals, x_labels)
        self.y_svr_gaze_estimator.fit(signals, y_labels)

    def draw_rect_with_text(self, text, text_color, width, height, rect_color=None, font_size=32, center=None,
                            left=0, top=0):
        """
        Creates a rectangle with the provided width and height at the provided center location. Draws the rectangle
        with rect_color if it is not None. Writes text with the desired font color at the center of the rectangle.
        The font size will shrink until it can fit within the rectangle in a single line.
        """
        while font_size > 2:
            font = self.fonts[font_size]
            req_width, req_height = font.size(text)
            if req_width <= width and req_height <= height:
                break
            font_size -= 2
        rect = pygame.Rect(left, top, width, height)
        if center is not None:
            rect.center = center
        if rect_color is None:
            rect_color = self.bg_color
        pygame.draw.rect(self.screen, rect_color, rect)
        text_img = font.render(text, True, text_color)
        if center is not None:
            text_rect = text_img.get_rect(center=center)
        else:
            text_rect = text_img.get_rect(left=left, top=top)
        self.screen.blit(text_img, text_rect)
        return text_rect

    def draw_circle_with_text(self, text, text_color, center, radius, circle_color=None, font_size=32):
        """
        Creates a circle with the provided width and height at the provided center location. Draws the circle
        with circle_color if it is not None. Writes text with the desired font color at the center of the circle.
        The font size will shrink until it can fit within the circle.
        """
        font = self.fonts[font_size]
        while font_size > 2:
            req_width, req_height = font.size(text)
            if req_width <= 2 * radius and req_height <= 2 * radius:
                break
            font_size -= 2
            font = self.fonts[font_size]
        center = np.round(center).astype(int)
        if circle_color is None:
            circle_color = self.bg_color
        gfxdraw.aacircle(self.screen, center[0], center[1], radius, circle_color)
        gfxdraw.filled_circle(self.screen, center[0], center[1], radius, circle_color)
        text_img = font.render(text, True, text_color)
        text_rect = text_img.get_rect(center=center)
        self.screen.blit(text_img, text_rect)

    def uncenter_coord(self, coord):
        return np.array([coord[0] + self.width / 2, self.height / 2 - coord[1]])

