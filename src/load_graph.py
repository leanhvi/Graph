from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from operator import add

def load_map_file(sc, filePath):
    rdd = sc.textFile(filePath)
    graph = rdd.map(process_line)
    return graph

def process_line(line):
    line = line.strip()
    ps = line.split(" ")
    node_id = ps[0]
    nb = ps[1:]
    return (node_id, nb)

def computeContribs(nbs, rank):
    """Calculates Node contributions to the rank of other Nodes."""
    num_nbs = len(nbs)
    for nb in nbs:
        yield (nb, rank / num_nbs)


def rank(it, nodes):
    ranks = nodes.map(lambda node: (node[0], 1.0))
    # Calculates and updates node ranks continuously using PageRank algorithm.
    for i in range(it):
        # Calculates Node contributions to the rank of other URLs.
        contribs = nodes.join(ranks).flatMap(lambda node: computeContribs(node[1][0], node[1][1]))
        # Re-calculates Node ranks based on neighbor contributions.
        ranks = contribs.reduceByKey(add).mapValues(lambda rank: rank * 0.85 + 0.15)
    # Collects all Node ranks and dump them to console.
    rank_nodes = ranks.collect()
    rank_nodes = sorted(rank_nodes, key=lambda a: a[1], reverse=True)
    for (node, rank) in rank_nodes:
        print("%s\t%s." % (node, rank))


if __name__ == "__main__":
    sc = SparkContext()
    spark = SparkSession(sc)
    filePath = "data/graph.txt"
    graph = load_map_file(sc, filePath)
    print(graph.collect())