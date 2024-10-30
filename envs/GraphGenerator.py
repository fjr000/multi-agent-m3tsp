import numpy as np


class GraphGenerator:
    def __init__(self, batch_size=2, num=5, dim=2, seed=None):
        self.dim = dim
        self.batch_size = batch_size
        self.num = num
        self.last_data = None
        self.last_distance_matrix = None
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

    def generate(self, batch_size=None, num=None, dim=None):
        B = self.batch_size if batch_size is None else batch_size
        N = self.num if num is None else num
        D = self.dim if dim is None else dim

        self.last_data = np.random.rand(B, N, D)
        self.last_distance_matrix = None
        return self.last_data

    def generate_distance_matrix(self, batch_size=None, num=None, dim=None):
        data = self.generate(batch_size, num, dim)
        self.last_distance_matrix = self.nodes_to_matrix(data)
        return self.last_distance_matrix
    @staticmethod
    def nodes_to_matrix(nodes):
        # coords 的维度为 [B, N, 2]
        # 使用广播机制计算欧几里得距离矩阵，形状 [B, N, N]

        # (B, N, 1, 2) 扩展成 (B, N, N, 2)
        coords_expanded_1 = np.expand_dims(nodes, axis=2)

        # (B, 1, N, 2) 扩展成 (B, N, N, 2)
        coords_expanded_2 = np.expand_dims(nodes, axis=1)

        # 使用广播机制计算欧几里得距离
        distance_matrix = np.sqrt(np.sum((coords_expanded_1 - coords_expanded_2) ** 2, axis=-1))

        return distance_matrix

    def __vio_nodes_to_matrix(self, nodes):
        matrix = np.zeros((nodes.shape[0], nodes.shape[1], nodes.shape[1]))
        for b, batch in enumerate(nodes):
            for i in range(nodes.shape[1]):
                for j in range(nodes.shape[1]):
                    if i == j:
                        matrix[b, i, j] = 0
                    else:
                        distance = np.sqrt((batch[i, 0] - batch[j, 0]) ** 2 +
                                           (batch[i, 1] - batch[j, 1]) ** 2)
                        # 存储在矩阵中
                        matrix[b, i, j] = distance
        return matrix

    def test(self):
        return self.generate(self.batch_size, self.num, self.dim)


if __name__ == '__main__':
    dg = GraphGenerator()
    data = dg.test()
    # print(f"generate data size:{data.shape}")
    # print(f"generate data:\n{data}")

    data = dg.generate(2, 3, 2)
    matrix = dg.nodes_to_matrix(data)
    print(np.all(dg.vio_nodes_to_matrix(data) == dg.nodes_to_matrix(data)))
    print(dg.generate_distance_matrix(2, 3, 2))
