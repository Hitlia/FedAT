import numpy as np
from collections import deque
from scipy.optimize import minimize


class SPOT:
    def log_prob(self, y: np.ndarray, gamma: np.ndarray, sigma_recip: np.ndarray):
        """
        计算联合概率密度。

        :param y: array :math:`Y_t = {X_i - t | X_i > t}` with shape :math:`(N_t,)`;
        :param gamma: array with shape :math:`(k,)`, k 为优化近似求解中点的个数;
        :param sigma_recip: sigma 的倒数, array with shape :math:`(k,)`.

        :return: array with shape :math:`(k,)`.
        """
        sample_num = y.shape[0]
        temp = np.log(1 + (gamma * sigma_recip).reshape(-1, 1) *
                      y.reshape(1, -1)).sum(axis=1)
        sigma_recip[sigma_recip == 0] = 1e-6
        ret = sample_num * np.log(sigma_recip) - (1 + 1 / gamma) * temp
        return ret

    def _compute_threshold(self, t, gamma, sigma, anomaly_ratio, nt, n):
        """计算 z_q"""
        temp = (anomaly_ratio * n / nt) ** (-gamma)
        return t + sigma / gamma * (temp - 1)

    def _grimshaw(self, y: np.ndarray, k=10, x0=None):
        """
        :param y: array :math:`Y_t = {X_i - t | X_i > t}` with shape :math:`(N_t,)`;
        :param k: 优化近似求解中点的个数;
        :param x0: 这 k 个点的初始值 with shape :math:`(k,)`.
        """
        def v(x: np.ndarray):
            return 1 + np.log(1 + x.reshape(-1, 1) * y.reshape(1, -1)).mean(axis=1)

        def optimize_func(x: np.ndarray):
            z = 1 + x.reshape(-1, 1) * y.reshape(1, -1)
            ux, vx = (1 / z).mean(axis=1), 1 + np.log(z).mean(axis=1)
            jac_u = -(y / np.square(z)).mean(axis=1)
            jac_v = (y / z).mean(axis=1)

            uv = ux * vx - 1
            target = np.square(uv).sum()
            jac = jac_u * vx + ux * jac_v
            return target, 2 * uv * jac

        if x0 is not None:
            assert isinstance(x0, np.ndarray) and x0.shape[0] == k
        else:
            x0 = np.zeros(k)

        low, high = -1 / y.max(), 2 * (y.mean() - y.min()) / np.square(y.min())
        mid = high * y.min() / y.mean()

        canditate_x = np.zeros(k)
        solution = minimize(
            optimize_func,
            x0=x0[:k // 2],
            method='L-BFGS-B', jac=True,
            bounds=np.array([low, 0]).reshape(1, -1).repeat(k // 2, axis=0)
        )
        canditate_x[:k // 2] = solution.x
        solution = minimize(
            optimize_func,
            x0=x0[-k // 2:],
            method='L-BFGS-B', jac=True,
            bounds=np.array([mid, high]).reshape(1, -1).repeat(k // 2, axis=0)
        )
        canditate_x[-k // 2:] = solution.x

        gamma = v(canditate_x) - 1
        gamma[gamma == 0] = 1e-6
        sigma_recip = canditate_x / gamma
        log_prob = self.log_prob(y, gamma, sigma_recip)

        target_index = np.argmax(log_prob)
        return gamma[target_index], 1 / sigma_recip[target_index], canditate_x

    def _pot(self, x: np.ndarray, anomaly_ratio: float, initial_thr_ratio=None):
        if initial_thr_ratio is None:
            initial_thr_ratio = anomaly_ratio * 3.5

        t = np.percentile(x, (1 - initial_thr_ratio) * 100, axis=0)
        y = x[x > t] - t
        gamma, sigma, x0 = self._grimshaw(y)

        threshold = self._compute_threshold(
            t, gamma, sigma, anomaly_ratio, y.shape[0], x.shape[0])

        return threshold, t, x0

    def spot(self, x: np.ndarray, anomaly_ratio: float, initial_seq_len=None,
             initial_thr_ratio=None, return_t=False):
        """
        :param x: 输入序列, array with shape :math:`(n,)`;
        :param anomaly_ratio: 异常率 q.
        :param initial_seq_len: 初始序列长度, SPOT 算法步骤 1 中的 n 值;
        :param initial_thr_ratio: 极值率，即极值点占整个序列的比例，用于确定初始阈值 t.
        """
        if initial_seq_len is None:
            initial_seq_len = int(x * 0.25)
        z_q, t, x0 = self._pot(x[:initial_seq_len], anomaly_ratio, initial_thr_ratio)

        count_sample, count_y = initial_seq_len, (x[:initial_seq_len] > t).sum()
        anomaly_indexes, y = [], x[x > t] - t
        for i in range(initial_seq_len, x.shape[0]):
            if x[i] > z_q:
                anomaly_indexes.append(i)
            elif x[i] > t:
                count_sample, count_y = count_sample + 1, count_y + 1
                gamma, sigma, x0 = self._grimshaw(y[:count_y], x0=x0)
                z_q = self._compute_threshold(
                    t, gamma, sigma, anomaly_ratio, count_y, count_sample)
            else:
                count_sample += 1

        if return_t:
            return anomaly_indexes, t
        return anomaly_indexes

    def dspot(self, x: np.ndarray, depth: int, anomaly_ratio: float,
              initial_seq_len=None, initial_thr_ratio=None):
        """
        :param depth: 初始窗口的长度，即 DSPOT 步骤 1 中的 d 值;
        """
        if initial_seq_len is None:
            initial_seq_len = int(len(x) * 0.25)

        mu, y = x[:depth].mean(), np.zeros(initial_seq_len - depth)
        for i in range(depth, initial_seq_len):
            y[i - depth] = x[i] - mu
            mu = mu + (x[i] - x[i - depth]) / depth

        z_q, t, x0 = self._pot(y, anomaly_ratio, initial_thr_ratio)

        count_sample, count_y = initial_seq_len - depth, (y > t).sum()
        mu_indexes = deque(range(initial_seq_len - depth, initial_seq_len))
        anomaly_indexes = []
        y = y[y > t] - t
        for i in range(initial_seq_len, x.shape[0]):
            cur_x = x[i] - mu
            if cur_x > z_q:
                anomaly_indexes.append(i)
                continue
            elif cur_x > t:
                count_sample, count_y = count_sample + 1, count_y + 1
                y = np.append(y, cur_x - t)
                gamma, sigma, x0 = self._grimshaw(y, x0=x0)
                z_q = self._compute_threshold(
                    t, gamma, sigma, anomaly_ratio, count_y, count_sample)
            else:
                count_sample += 1

            mu = mu + (x[i] - x[mu_indexes.popleft()]) / depth
            mu_indexes.append(i)

        return anomaly_indexes