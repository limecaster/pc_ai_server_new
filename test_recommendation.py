import requests
import json
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_recommendation")

# API URL
BASE_URL = "http://127.0.0.1:3003"

def test_health():
    """Test the health endpoint"""
    try:
        logger.info("Testing health endpoint")
        response = requests.get(f"{BASE_URL}/api/recommendations/health")
        response.raise_for_status()
        data = response.json()
        logger.info(f"Health response: {json.dumps(data, indent=2)}")
        return True
    except Exception as e:
        logger.error(f"Error testing health endpoint: {str(e)}")
        return False

def test_train():
    """Test the train endpoint"""
    try:
        logger.info("Testing train endpoint")
        response = requests.post(f"{BASE_URL}/api/recommendations/train")
        response.raise_for_status()
        data = response.json()
        logger.info(f"Train response: {json.dumps(data, indent=2)}")
        return True
    except Exception as e:
        logger.error(f"Error testing train endpoint: {str(e)}")
        return False

def test_recommendations(product_id):
    """Test the recommendations endpoint"""
    try:
        logger.info(f"Testing recommendations for product {product_id}")
        response = requests.get(
            f"{BASE_URL}/api/recommendations",
            params={"productId": product_id}
        )
        response.raise_for_status()
        data = response.json()
        logger.info(f"Recommendations response: {json.dumps(data, indent=2)}")
        return True
    except Exception as e:
        logger.error(f"Error testing recommendations endpoint: {str(e)}")
        return False

def wait_for_server(max_attempts=5, interval=2):
    """Wait for server to be available"""
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{BASE_URL}/api/recommendations/health")
            if response.status_code == 200:
                logger.info("Server is available")
                return True
        except:
            pass
        
        logger.info(f"Server not available yet. Waiting {interval} seconds...")
        time.sleep(interval)
    
    logger.error(f"Server not available after {max_attempts} attempts")
    return False

if __name__ == "__main__":
    # Wait for server to be available
    if not wait_for_server():
        logger.error("Cannot connect to recommendation server. Make sure it's running.")
        exit(1)
    
    # Test health endpoint
    if not test_health():
        logger.error("Health check failed.")
        exit(1)
    
    # Test train endpoint
    if not test_train():
        logger.error("Training test failed.")
        exit(1)
    
    # Wait for training to complete
    logger.info("Waiting for training to complete...")
    time.sleep(5)
    
    # Test with a sample product ID (replace with a valid ID from your database)
    test_recommendations("sample-product-id")
    
    logger.info("All tests completed.") 