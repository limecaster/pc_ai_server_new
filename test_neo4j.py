from neo4j import GraphDatabase

def test_connection():
    # Connect to Neo4j
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', '12345678'))
    
    # Run a query to count nodes by label
    with driver.session() as session:
        print('Connected to Neo4j successfully!')
        print('\nNode count by label:')
        result = session.run('MATCH (n) RETURN labels(n) as labels, count(n) as count LIMIT 10')
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

if __name__ == "__main__":
    try:
        test_connection()
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}") 