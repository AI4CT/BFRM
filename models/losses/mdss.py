import numpy as np

class MDSS:
    def __init__(self):
        self.K1 = 0.01
        self.K2 = 0.03

    def _calculate_ssim_components(self, mu1, mu2, sigma1, sigma2, cov):
        C1 = (self.K1 * self.len)**2
        C2 = (self.K2 * self.len)**2
        C3 = C2 / 2

        l = (2 * mu1 * mu2 + C1) / (mu1**2 + mu2**2 + C1)
        c = (2 * sigma1 * sigma2 + C2) / (sigma1**2 + sigma2**2 + C2)
        s = (cov + C3) / (sigma1 * sigma2 + C3)

        return l, c, s

    def _ssim_index(self, l, c, s, alpha=1, beta=1, gamma=1):
        return l**alpha * c**beta * s**gamma

    def _calculate_3d_ssim(self, batch_data1, batch_data2):
        self.len = batch_data1.shape[0]
        ssim_x, ssim_y, ssim_z, ssim_3d = np.array([]), np.array([]), np.array([]), np.array([])

        for data1, data2 in zip(batch_data1, batch_data2):

            mu1_x = np.mean(data1[:, 0])
            mu2_x = np.mean(data2[:, 0])
            sigma1_x = np.std(data1[:, 0], ddof=1)
            sigma2_x = np.std(data2[:, 0], ddof=1)
            cov_x = np.cov(data1[:, 0], data2[:, 0], ddof=1)[0][1]

            mu1_y = np.mean(data1[:, 1])
            mu2_y = np.mean(data2[:, 1])
            sigma1_y = np.std(data1[:, 1], ddof=1)
            sigma2_y = np.std(data2[:, 1], ddof=1)
            cov_y = np.cov(data1[:, 1], data2[:, 1], ddof=1)[0][1]

            mu1_z = np.mean(data1[:, 2])
            mu2_z = np.mean(data2[:, 2])
            sigma1_z = np.std(data1[:, 2], ddof=1)
            sigma2_z = np.std(data2[:, 2], ddof=1)
            cov_z = np.cov(data1[:, 2], data2[:, 2], ddof=1)[0][1]

            l_x, c_x, s_x = self._calculate_ssim_components(mu1_x, mu2_x, sigma1_x, sigma2_x, cov_x)
            l_y, c_y, s_y = self._calculate_ssim_components(mu1_y, mu2_y, sigma1_y, sigma2_y, cov_y)
            l_z, c_z, s_z = self._calculate_ssim_components(mu1_z, mu2_z, sigma1_z, sigma2_z, cov_z)

            ssim_x = np.append(ssim_x, self._ssim_index(l_x, c_x, s_x))
            ssim_y = np.append(ssim_y, self._ssim_index(l_y, c_y, s_y))
            ssim_z = np.append(ssim_z, self._ssim_index(l_z, c_z, s_z))
            rho, sigma, omega = 1, 1, 1
            ssim_3d = np.append(ssim_3d, ssim_x[-1]**rho * ssim_y[-1]**sigma * ssim_z[-1]**omega)

        return ssim_x, ssim_y, ssim_z, ssim_3d

    def evaluate(self, batch_data1, batch_data2):
        assert batch_data1.shape == batch_data2.shape, "Input data must have the same shape"
        assert len(batch_data1.shape) == 3 and batch_data1.shape[2] == 3, "Input data must be 3-dimensional with 3 channels (x, y, z)"

        return self._calculate_3d_ssim(batch_data1, batch_data2)

if __name__ == "__main__":
    batch_data1 = np.random.rand(10, 100, 3)  # 10 batches of 100 points with x, y, z coordinates
    batch_data2 = np.random.rand(10, 100, 3)
    mdss = MDSS()
    x_ssim, y_ssim, z_ssim, three_d_ssim = mdss.evaluate(batch_data1, batch_data2)
    print(x_ssim, y_ssim, z_ssim, three_d_ssim)