import numpy as np

class RunningMeanStd:
	def __init__(self, epsilon=1, shape=()):
		"""
		Calulates the running mean and std of a data stream
		https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
		:param epsilon: helps with arithmetic issues
		:param shape: the shape of the data stream's output
		"""
		self.mean = np.zeros(shape)
		self.var = np.ones(shape)
		self.count = epsilon

	def update(self, arr):
		batch_mean = np.mean(arr, axis=0)
		batch_var = np.var(arr, axis=0)
		batch_count = arr.shape[0]
		self.update_from_moments(batch_mean, batch_var, batch_count)

	def update_from_moments(self, batch_mean, batch_var, batch_count):
		delta = batch_mean - self.mean
		tot_count = self.count + batch_count

		new_mean = self.mean + delta * batch_count / tot_count
		m_a = self.var * self.count
		m_b = batch_var * batch_count
		m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
		new_var = m_2 / (self.count + batch_count)

		new_count = batch_count + self.count

		self.mean = new_mean
		self.var = new_var
		self.count = new_count

def run_bc_batch(replay_buffer, policy, batch_size):
	batch = get_batch_from_buffer(replay_buffer, batch_size)
	o = batch["observations"]
	u = batch["actions"]
	og = o
	dist = policy(og)
	pred_u, log_pi = dist.rsample_and_logprob()
	stats = dist.get_diagnostics()

	mse = (pred_u - u) ** 2
	mse_loss = mse.mean()

	policy_logpp = dist.log_prob(u, )
	logp_loss = -policy_logpp.mean()
	policy_loss = logp_loss
