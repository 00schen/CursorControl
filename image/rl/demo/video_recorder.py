import os

from gym.wrappers.monitoring import video_recorder

def video_factory(base):
	class VideoRecorder(base):
		"""
		It requires ffmpeg or avconv to be installed on the machine.
		:param video_path: (str) Where to save videos
		:param video_length: (int)  Length of recorded videos
		:param name_prefix: (str) Prefix to the video name
		"""

		def __init__(self,config):
			super().__init__(config)

			self.video_recorder = None

			video_config = config.pop('video_config')
			self.video_path = os.path.abspath(video_config['video_path'])
			# Create output folder if needed
			os.makedirs(self.video_path, exist_ok=True)

			self.video_name = video_config['video_name']
			self.video_episodes = video_config['video_episodes']

			self.recorded_episodes = 0
			self.metadata = {
				"render.modes":['human','rgb_array']
			}

		def reset(self):
			obs = super().reset()
			if self.recorded_episodes == 0:
				self.start_video_recorder()
			elif self.recorded_episodes >= self.video_episodes:
				self.video_recorder.close()
			self.recorded_episodes += 1
			
			return obs

		def start_video_recorder(self):
			video_name = self.video_name
			base_path = os.path.join(self.video_path, video_name)
			self.video_recorder = video_recorder.VideoRecorder(
				env=self, base_path=base_path
			)
			print("video starting")

			self.video_recorder.capture_frame()

		def step(self,action):
			obs, r, done, info = super().step(action)
			self.video_recorder.capture_frame()
			if not self.video_recorder.functional:
				print("video recorder broken")		

			return obs, r, done, info

		def __del__(self):
			self.close()
	return VideoRecorder