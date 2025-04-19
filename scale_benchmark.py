from neo4j import GraphDatabase

# Neo4j connection details
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "12345678")

# Connect to Neo4j
driver = GraphDatabase.driver(URI, auth=AUTH)

def get_labels(tx):
    result = tx.run("CALL db.labels() YIELD label RETURN label")
    return [record["label"] for record in result]

def update_benchmark_score(tx, label):
    query = f"""
    MATCH (n:{label})
    WITH max(n.benchmarkScore) AS maxScore
    WHERE maxScore > 0
    MATCH (n:{label})
    SET n.benchmarkScore = (n.benchmarkScore * 100.0) / maxScore
    """
    tx.run(query)

# Execute queries
with driver.session() as session:
    labels = session.execute_read(get_labels)
    for label in labels:
        session.execute_write(update_benchmark_score, label)

# Close connection
driver.close()
