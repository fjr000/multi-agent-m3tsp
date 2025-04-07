import os.path
import pickle

import numpy as np
import tsplib95
from sympy.external.tests.test_numpy import array

result_dict={
    "eil51":{
        2:(222.73, 222.73),
        3:(150.70, 159.57),
        5:(96.91, 123.96),
        7:(75.91, 112.07)
    },

    "berlin52":{
        2: (4049.05, 4110.21),
        3: (2753.63, 3244.37),
        5: (1671.69, 2441.39),
        7: (1272.06, 2440.92)
    },

    "eil76": {
        2: (280.85, 280.85),
        3: (186.34, 197.34),
        5: (117.61, 150.30),
        7: (88.35, 139.62)
    },

    "rat99": {
        2: (620.99, 728.71),
        3: (426.25, 587.17),
        5: (271.91, 469.25),
        7: (210.41, 443.91)
    },

}

norm_result_dict_LKH3={
    "eil51":{
        2:3.2527464957200003,
        3:2.31265686438,
        5:1.71210844726,
        7:1.62422970354
    },

    "berlin52":{
        2: 2.36723471673,
        3: 1.7886253008900002,
        5: 1.40282673412,
        7: 1.40282685826
    },

    "eil76": {
        2: 3.77864752357,
        3: 2.640707914,
        5: 1.88967938932,
        7: 1.678441131013
    },

    "rat99": {
        2: 3.1062462062800003,
        3: 2.3915171473099996,
        5: 2.112721027597,
        7: 2.01637138933
    },

}


class TspInstanceFileTool(object):
    def __init__(self):
        pass

    @staticmethod
    def writeTSPInstanceToFile(filename, graph, depot = (1,), instance_name = None, scale = 100000):
        assert len(graph.shape) == 2 and graph.shape[1] == 2, "graph size should match [N,2]"

        dim = graph.shape[0]
        if instance_name is None:
            instance_name = f"TSP_{dim}"
        with open(filename, 'w') as f:
            instance_name = f"NAME : {instance_name}" +"\n"
            type = "TYPE : TSP" + "\n"
            dim = f"DIMENSION : {graph.shape[0]}" + "\n"
            distance_methods = "EDGE_WEIGHT_TYPE: EUC_2D" + "\n"
            f.write(instance_name)
            f.write(type)
            f.write(dim)
            f.write(distance_methods)
            f.write("NODE_COORD_SECTION\n")
            for i in range(graph.shape[0]):
                f.write(f"{i+1} {graph[i, 0] * scale} {graph[i, 1]*scale}\n")
            f.write("\nDEPOT_SECTION\n")
            for dp in depot:
                f.write(f"{dp}\n")
            f.write("-1\n")
            f.write("EOF")

    @staticmethod
    def readTSPInstanceFromFile(filename, scale = 100000):
        try:
            with open(filename, 'r') as f:
                line = f.readline().strip()
                while not line.startswith("DIMENSION") :
                    line = f.readline().strip()
                dim_list = line.split(":")
                dim = eval(dim_list[1].strip())
                graph = np.zeros((dim,2))
                depots = []

                line = f.readline()
                while line.find("NODE_COORD_SECTION") == -1:
                    line = f.readline()
                line = f.readline().strip()
                while not line.startswith("DEPOT_SECTION") and not line.startswith("EOF"):
                    if not line:
                        line = f.readline().strip()
                        continue
                    cord = line.split(" ")
                    cord = [eval(c) for c in cord]
                    graph[cord[0]-1,0] = float(cord[1]) / scale
                    graph[cord[0]-1,1] = float(cord[2]) / scale
                    line = f.readline().strip()


                if line.startswith("DEPOT_SECTION"):
                    line = f.readline().strip()
                    while not line.startswith("-1") and not line.startswith("EOF"):
                        if not line:
                            line = f.readline().strip()
                            continue
                        depots.append(int(line))
                        line = f.readline().strip()
                return graph, depots

        except FileNotFoundError:
            print(f"File \"{filename}\" not found")

    @staticmethod
    def writeLKH3MTSPPar(filename,tsp_filename, salesmen = 1, depot = 1, output_filename=None, time_limit = 60):
        if output_filename is None:
            output_filename = tsp_filename.split(".")[-2] + ".tour"
        with open(filename, 'w') as f:
            f.write("SPECIAL\n")
            f.write(f"PROBLEM_FILE = {tsp_filename}\n")
            f.write("STOP_AT_OPTIMUM = NO\n")
            f.write("RUNS = 1\n")
            f.write("SEED = 0\n")
            f.write(f"SALESMEN = {salesmen}\n")
            f.write(f"DEPOT = {depot}\n")
            f.write(f"INITIAL_TOUR_ALGORITHM = MTSP\n")
            f.write("MTSP_OBJECTIVE = MINMAX\n")
            f.write(f"TIME_LIMIT = {time_limit}\n")
            f.write(f"MTSP_SOLUTION_FILE = {output_filename}\n")


    @staticmethod
    def readLKH3Route(filename, scale = 100_000):
        with open(filename, 'r') as f:
            cost_line = f.readline().strip()
            num_line = f.readline().strip()
            cost_line = cost_line.replace('_','.')
            min_cost = eval(cost_line.split(':')[1]) / scale
            salesmen = eval(num_line.split(' ')[5])
            trajectory = [[]for i in range(salesmen)]
            for idx, line in enumerate(f.readlines()):
                if idx >= salesmen:
                    break
                line = line.strip()
                line_split = line.split(" ")
                for pos in line_split:
                    if not pos.startswith("("):
                        trajectory[idx].append(eval(pos))
                    else:
                        break

        return min_cost, salesmen, trajectory

    @staticmethod
    def loadTSPLib(graph_dir:str, instance_name:str):
        assert instance_name in ("eil51", "berlin52", "eil76", "rat99"), "not support instance"
        p = tsplib95.load_problem(os.path.join(graph_dir,instance_name)+".tsp")
        n = p.dimension
        graph = np.zeros((n, 2), dtype=np.float32)
        max_V = 0
        for idx, node in enumerate(p.node_coords.items()):
            graph[idx, 0] = node[1][0]
            graph[idx, 1] = node[1][1]
            max_V = max(max_V, graph[idx, 0])
            max_V = max(max_V, graph[idx, 1])

        norm_graph = graph / max_V
        return norm_graph[None,:], max_V

    @staticmethod
    def saveDataSet(filename, data:np.ndarray):
        filedir = os.path.split(filename)[0]

        if not os.path.isdir(filedir):
            os.makedirs(filedir)

        np.savez_compressed(filename + '.npz', data)
        return filename + '.npz'

    @staticmethod
    def loadDataSet(filename:str):
        graph = np.load(filename)
        return graph

if __name__ == "__main__":
    import envs.GraphGenerator as GG
    gg = GG.GraphGenerator()
    graph = gg.generate(1,50,2)
    graph = graph[0]
    depots = (1,)
    filename = "TSP.tsp"
    # TspInstanceFileTool.writeTSPInstanceToFile(filename, graph, depot = depots, instance_name = "TSP")

    load_graph, load_depots = TspInstanceFileTool.readTSPInstanceFromFile(filename)
    import algorithm.OR_Tools.mtsp as ORT
    # a = ORT.ortools_solve_mtsp(load_graph[np.newaxis,],5,100000)
    # print(a[1])
    from utils.GraphPlot import GraphPlot as GP
    gp = GP()
    # gp.draw_route(load_graph[np.newaxis,], a[0], title=f"or_tools_cost:{a[1]}_time:{a[2]}")
    #
    # for p,l in zip(graph,load_graph):
    #     for d,dd in zip(p,l):
    #         if not np.isclose(d,dd):
    #             assert False
    #
    # for d,dd in zip(depots,load_depots):
    #     if d != dd:
    #         assert False

    TspInstanceFileTool.writeLKH3MTSPPar("TSP.par",filename,5,1,"TSP_route.par")
    min_cost, salesmen, trajectory = TspInstanceFileTool.readLKH3Route("TSP_route.par")

    gp.draw_route(load_graph[np.newaxis,], trajectory, title=f"LKH3:{min_cost}_time:?", one_first=True)