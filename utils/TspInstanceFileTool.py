import numpy as np


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




if __name__ == "__main__":
    import envs.GraphGenerator as GG
    gg = GG.GraphGenerator()
    graph = gg.generate(1,100,2)
    graph = graph[0]
    depots = (1,2,3,4)
    filename = "../envs/Instance/TSP.tsp"
    TspInstanceFileTool.writeTSPInstanceToFile(filename, graph, depot = depots, instance_name = "TSP")

    load_graph, load_depots = TspInstanceFileTool.readTSPInstanceFromFile(filename)

    for p,l in zip(graph,load_graph):
        for d,dd in zip(p,l):
            if not np.isclose(d,dd):
                assert False

    for d,dd in zip(depots,load_depots):
        if d != dd:
            assert False

