import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.decomposition import PCA

import numpy as np


def is_point_in_ellipsoid(point, center, axes, radii):
    point = np.array(point) - np.array(center)

    projections = [np.dot(point, axis) for axis in axes]

    distance_squared = sum((p / (r + 0.00001)) ** 2 for p, r in zip(projections, radii))

    return distance_squared <= 1


def project_point_to_plane(point, vectors):
    projection = np.zeros_like(point)
    for v in vectors:
        projection += np.dot(point, v) * v
    return projection


class PCAEllipsoid:
    def __init__(self, data, n_components=None, n_std=2.0):
        self.n_std = n_std
        self.pca = PCA(n_components=n_components)
        self.pca.fit(data)
        self.chi2_val = chi2.ppf((1 - 2 * (1 - 0.5 ** (1 / n_std))), data.shape[1])

    def contains(self, sample):
        transformed_sample = self.pca.transform(sample.reshape(1, -1))
        d = mahalanobis(transformed_sample[0], np.zeros(transformed_sample.shape[1]),
                        np.diag(self.pca.explained_variance_))
        return d < self.chi2_val

    def project(self, center, sample, k):
        vectors = self.pca.components_[:k]
        dir = sample - center
        dim = len(center)
        project = np.zeros(dim)
        cnt = 0
        for v in self.pca.components_:
            cnt += 1
            project += np.dot(v, dir) * v
            if cnt == k:
                break
        return project + center

    def get_shape(self):
        # 特征值（长度）
        lengths = np.sqrt(self.pca.explained_variance_) * 2
        # 特征向量（方向）
        directions = self.pca.components_
        center = self.pca.mean_
        return lengths, directions, center


def VineLikeExpansion(obs_points, free_points, q_near, q_rands, dim=3, output=False):
    '''
    all is np. Array
    :param obs_points:
    :param free_points:
    :param q_near:
    :param dim:
    :return: list of extend points, in form of list
    '''
    obs_elipsoid = PCAEllipsoid(obs_points)

    obslen, obsaxes, obscenter = obs_elipsoid.get_shape()

    if output:
        lens, axes, center = obs_elipsoid.get_shape()
        print("obsepl")
        print(lens)
        print(axes)
        print(center)

    tendril_set = []

    for free_sample in free_points:
        if is_point_in_ellipsoid(free_sample, obscenter, obsaxes, obslen):
            tendril_set.append(free_sample)

    q_projects = []
    for q_rand in q_rands:
        q_projects.append(obs_elipsoid.project(q_near, q_rand, dim - 1))
    if len(tendril_set) >= 2:
        q_2 = np.zeros(dim)
        for tendril_p in tendril_set:
            q_2 = q_2 + tendril_p
        q_2 = q_2 / (len(tendril_set))
        if output:
            print("tendril")
            print(tendril_set)
            print("q_2 is ")
            print(q_2)
        if obs_elipsoid.contains(q_near):
            free_elipsoid = PCAEllipsoid(free_points)
            extend_q = []
            for q_rand in q_rands:
                extend_q.append(free_elipsoid.project(q_near, q_rand, dim - 1))

            # print("inside narrow passage ")
            for q_pro in q_projects:
                extend_q.append(q_pro)
            extend_q.append(q_2)
            return extend_q, 0

        q_projects.append(q_2)
        return q_projects, 1
    else:
        return q_projects, 2


if __name__ == '__main__':
    # unit test1
    Obs_points = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [1, 1.2], [2.3, 2], [3.5, 3]])
    Free_points = np.array([[2, 2], [2.1, 2.2]])
    q_near = np.array([0, 0.1])
    q_rand = np.array([2.0, 2.0])
    extend, case = VineLikeExpansion(Obs_points, Free_points, q_near, [q_rand], output=True)
    print(extend)
    assert case == 1;
