from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Neo4jRetriever:
    def __init__(self, uri=None, user=None, password=None):
        # Use provided parameters or environment variables
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://pc_neo4j:7687')
        self.user = user or os.getenv('NEO4J_USER', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', '12345678')
        
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        self.driver.close()

    def search_product(self, product_name):
        """Find the most relevant product using full-text search."""
        query = """
        CALL db.index.fulltext.queryNodes('searchProductName', $name) YIELD node, score
        RETURN node.name AS product_name, labels(node) AS type
        ORDER BY score DESC LIMIT 1
        """
        with self.driver.session() as session:
            result = session.run(query, name=product_name)
            return result.single()

    def get_compatibility(self, product_name):
        """Find all compatible parts for the given product."""
        query = """
        MATCH (p {name: $name})-[:COMPATIBLE_WITH]->(compatible)
        RETURN compatible.name AS compatible_product, labels(compatible) AS type
        """
        with self.driver.session() as session:
            results = session.run(query, name=product_name)
            return [{"name": record["compatible_product"], "type": record["type"]} for record in results]

    def get_compatibility_response(self, user_query):
        """Get a formatted response about compatibility based on user query."""
        # Simple implementation - this could be enhanced with NLP
        try:
            product = self.search_product(user_query)
            if not product:
                return {
                    "type": "text",
                    "data": "Không tìm thấy sản phẩm phù hợp với truy vấn của bạn."
                }
                
            product_name = product["product_name"]
            compatible_parts = self.get_compatibility(product_name)
            
            if not compatible_parts:
                return {
                    "type": "text", 
                    "data": f"Không tìm thấy thông tin về tương thích cho {product_name}."
                }
                
            return {
                "type": "compatibility",
                "data": {
                    "product": product_name,
                    "compatible_parts": compatible_parts
                }
            }
        except Exception as e:
            print(f"Error in get_compatibility_response: {str(e)}")
            return {
                "type": "error",
                "data": "Có lỗi khi truy vấn thông tin tương thích. Vui lòng thử lại sau."
            }
