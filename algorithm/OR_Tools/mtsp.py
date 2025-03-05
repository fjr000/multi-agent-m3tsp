import time

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
from envs.GraphGenerator import GraphGenerator as GG
from utils.GraphPlot import GraphPlot as GP


def get_indexs(M, manager, routing, solution):
    max_route_distance = 0
    result = [[] for i in range(M)]
    for i in range(M):
        index = routing.Start(i)
        route_distance = 0
        while not routing.IsEnd(index):
            result[i].append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, i)
        result[i].append(manager.IndexToNode(index))
        max_route_distance = max(route_distance, max_route_distance)
    return result, max_route_distance



def ortools_solve_mtsp(graph, M = 5, C = 100000):
    N = 0
    if len(graph.shape) == 3:
        N = graph.shape[1]
    elif len(graph.shape) == 2:
        N = graph.shape[0]
    else:
        raise ValueError('Graph must be either 3 or 2 dimensional')
    distance_matrix = (C * GG.nodes_to_matrix(graph).squeeze(0)).astype(np.int32)
    manager = pywrapcp.RoutingIndexManager(N, M, 0)
    routing = pywrapcp.RoutingModel(manager)
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        dist = distance_matrix[from_node][to_node]
        # print(f"dist{from_node}->{to_node}:{dist}")
        return dist

    # 注册回调函数到 RoutingModel
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # 定义每一步的成本为距离
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 设置车辆的最大距离约束（可选，设置一个大值让车辆可以遍历所有城市）
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        C * N // M,  # max distance per vehicle
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    # 设置最大路径长度的系数，启用 Min-Max 目标

    distance_dimension.SetGlobalSpanCostCoefficient(C * C * N // M)

    # 设置搜索参数
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.time_limit.seconds = 60  # 限制为60秒
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # 求解问题
    t1 = time.time_ns()
    solution = routing.SolveWithParameters(search_parameters)
    t2 = time.time_ns()

    # 打印解决方案
    if solution:
        indexs, max_dist = get_indexs(M, manager, routing, solution)
        return indexs, max_dist / C, (t2 - t1) / 1e9
    else:
        return None, None, None


if __name__ == '__main__':
    B = 1
    N = 50
    D = 2
    M = 5
    C = 100000
    graph_generator = GG(B, N, D)
    graph_plot = GP()
    graph = graph_generator.generate(B,N,D)
    indexs, cost, used_time = ortools_solve_mtsp(graph, M, C)
    if indexs is not None:
        graph_plot.draw_route(graph, indexs, title=f"or_tools_cost:{cost}_time:{used_time}")
    else:
        print('No solution found !')

