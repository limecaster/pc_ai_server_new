from neo4j import GraphDatabase

def test_connection():
    # Connect to Neo4j
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', '12345678'))
    
    # Run a query to count nodes by label
    with driver.session() as session:
        print('Connected to Neo4j successfully!')
        print('\nNode count by label:')
        result = session.run('MATCH (n) RETURN labels(n) as labels, count(n) as count')
        for record in result:
            print(f'{record["labels"]}: {record["count"]} nodes')
    
    # Get database info
    with driver.session() as session:
        print('\nDatabase Info:')
        result = session.run('CALL dbms.components() YIELD name, versions, edition')
        for record in result:
            print(f'Name: {record["name"]}, Version: {record["versions"][0]}, Edition: {record["edition"]}')
    
    # List all node labels
    with driver.session() as session:
        print('\nAll Node Labels:')
        result = session.run('CALL db.labels()')
        labels = [record["label"] for record in result]
        print(', '.join(labels) if labels else "No labels found")
    
    # Close the driver connection
    driver.close()

def clear_label(label: str):
    """Clear all nodes with a specific label."""
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', '12345678'))
    with driver.session() as session:
        # Use Cypher label syntax for correct label matching
        session.run(f'MATCH (n:`{label}`) DELETE n')
    driver.close()

def set_neo4j_price(name:str, label:str, price:any):
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', '12345678'))
    with driver.session() as session:
        session.run(f'MATCH (n:{label}) WHERE n.name = "{name}" SET n.price = 0 RETURN n')
        result = session.run(f'MATCH (n:{label}) WHERE n.name = "{name}" RETURN n.price')
        print(f'Price set for {name}: {result.single()["n.price"]}')
    driver.close()

def get_neo4j_price(name:str, label:str):
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', '12345678'))
    with driver.session() as session:
        result = session.run(f'MATCH (n:{label}) WHERE n.name="{name}" RETURN n.price')
        return result.single()["n.price"]
    driver.close()

def validate_neo4j_label_price(label:str):
    """Validate that the price of a label is set correctly."""
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', '12345678'))
    with driver.session() as session:
        result = session.run(f'MATCH (n:{label}) RETURN n.price')
        # return all prices
        return [record["n.price"] for record in result.data()]
    driver.close()

if __name__ == "__main__":
    try:
        validate_neo4j_label_price("Motherboard")
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")