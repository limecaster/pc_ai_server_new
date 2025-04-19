from fastapi import FastAPI, Query, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
import pandas as pd
import os
import uvicorn
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from recommendation_service import RecommendationModel
import traceback
from dotenv import load_dotenv
import pathlib

# Load the appropriate environment file based on NODE_ENV
node_env = os.environ.get('NODE_ENV', 'development')
env_file = f".env.{node_env}" if node_env != 'development' else ".env.local"
env_path = pathlib.Path(__file__).parent / env_file

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("recommendation_api")

app = FastAPI(
    title="PC Recommendation API",
    description="API for product recommendations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models - one for standard product-based recommendations, one for advanced
standard_model = RecommendationModel(model_type="standard", model_path="models/standard_recommendation_model.pkl")
advanced_model = RecommendationModel(model_type="advanced", model_path="models/advanced_recommendation_model.pkl")

# Response models
class TrainingResponse(BaseModel):
    success: bool
    message: str

class Recommendation(BaseModel):
    recommendations: List[str]
    success: bool
    count: Optional[int] = None
    message: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    standard_model_status: str
    advanced_model_status: str
    product_count: int

# Database connection
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.environ.get('POSTGRES_HOST', 'localhost'),
            database=os.environ.get('POSTGRES_NAME', 'pc_ecommerce'),
            user=os.environ.get('POSTGRES_USER', 'postgres'),
            password=os.environ.get('POSTGRES_PASSWORD', 'postgres'),
            port=os.environ.get('POSTGRES_PORT', '5432')
        )
        return conn
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

# Function to train models in background
def train_models_task():
    try:
        conn = None
        try:
            conn = get_db_connection()
            
            # Fetch product data (needed for both models)
            product_query = """
            SELECT id, name, category, price::float, description
            FROM "Products"
            WHERE status = 'active'
            """
            product_data = pd.read_sql(product_query, conn)
            
            # Train standard model
            empty_behavior_data = pd.DataFrame(columns=['customer_id', 'session_id', 'product_id', 'event_type', 'created_at'])
            standard_model.train(empty_behavior_data, product_data)
            
            # Fetch comprehensive user behavior data for advanced model
            behavior_query = """
            SELECT customer_id, session_id, entity_id as product_id, event_type, created_at
            FROM "User_Behavior"
            WHERE event_type IN ('product_viewed', 'product_click', 'product_added_to_cart', 'order_created', 'payment_completed')
            AND entity_type = 'product'
            AND created_at > NOW() - INTERVAL '30 days'
            """
            behavior_data = pd.read_sql(behavior_query, conn)
            
            viewed_products_query = """
            SELECT customer_id, product_id, viewed_at as created_at
            FROM "Viewed_Products"
            WHERE viewed_at > NOW() - INTERVAL '30 days'
            """
            viewed_products_data = pd.read_sql(viewed_products_query, conn)
            
            order_query = """
            SELECT 
                o.customer_id, 
                od.product_id, 
                'order_purchased' as event_type, 
                o.created_at
            FROM "Orders" o
            JOIN "Order_Detail" od ON o.id = od.order_id
            WHERE o.status IN ('completed', 'delivered', 'processing')
            AND o.created_at > NOW() - INTERVAL '60 days'
            """
            order_data = pd.read_sql(order_query, conn)
            
            order_events_query = """
            SELECT 
                ub.customer_id,
                ub.session_id,
                p.id as product_id,
                ub.event_type,
                ub.created_at
            FROM "User_Behavior" ub
            JOIN "Orders" o ON ub.entity_id = o.id::text
            JOIN "Order_Detail" od ON o.id = od.order_id
            JOIN "Products" p ON od.product_id = p.id
            WHERE ub.entity_type IN ('order', 'payment')
            AND ub.event_type IN ('order_created', 'payment_completed')
            AND ub.created_at > NOW() - INTERVAL '60 days'
            """
            order_events_data = pd.read_sql(order_events_query, conn)
            
            # Combine all behavior data
            all_behavior_data = []
            
            if not behavior_data.empty:
                all_behavior_data.append(behavior_data)
            
            if not viewed_products_data.empty:
                viewed_as_behavior = pd.DataFrame({
                    'customer_id': viewed_products_data['customer_id'],
                    'session_id': None,
                    'product_id': viewed_products_data['product_id'],
                    'event_type': 'product_viewed',
                    'created_at': viewed_products_data['created_at']
                })
                all_behavior_data.append(viewed_as_behavior)
            
            if not order_data.empty:
                order_as_behavior = pd.DataFrame({
                    'customer_id': order_data['customer_id'],
                    'session_id': None,
                    'product_id': order_data['product_id'],
                    'event_type': order_data['event_type'],
                    'created_at': order_data['created_at']
                })
                all_behavior_data.append(order_as_behavior)
            
            if not order_events_data.empty:
                all_behavior_data.append(order_events_data)
            
            # Combine all behavior data
            if all_behavior_data:
                merged_behavior = pd.concat(all_behavior_data, ignore_index=True)
                
                # Train advanced model with comprehensive behavior data
                advanced_model.train(merged_behavior, product_data)
                
        finally:
            if conn:
                conn.close()
    
    except Exception as e:
        logger.error(traceback.format_exc())

# Train model endpoint
@app.post("/api/recommendations/train", response_model=TrainingResponse)
async def train_models(background_tasks: BackgroundTasks):
    try:
        # Add model training to background tasks
        background_tasks.add_task(train_models_task)
        
        return {
            "success": True, 
            "message": "Model training started in background for both standard and advanced models"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get standard recommendations endpoint (product-based)
@app.get("/api/recommendations", response_model=Recommendation)
async def get_recommendations(
    productId: Optional[str] = Query(None, description="Product ID to get recommendations for"),
    category: Optional[str] = Query(None, description="Optional category filter"),
    limit: int = Query(4, description="Maximum number of recommendations to return")
):
    try:
        if not productId:
            return {
                "success": False,
                "recommendations": [],
                "message": "Product ID is required for standard recommendations"
            }
        
        # Get recommendations from standard model
        recommendations = standard_model.get_recommendations(productId, category, limit)
        
        if not recommendations:
            return {
                "success": True,
                "recommendations": [],
                "message": "No recommendations available for the given parameters"
            }
        
        return {
            "success": True,
            "recommendations": recommendations,
            "count": len(recommendations)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Adding another endpoint with the same implementation for frontend compatibility
@app.get("/api/products/recommendations", response_model=Recommendation)
async def get_product_recommendations(
    productId: Optional[str] = Query(None, description="Product ID to get recommendations for"),
    category: Optional[str] = Query(None, description="Optional category filter"),
    limit: int = Query(4, description="Maximum number of recommendations to return")
):
    return await get_recommendations(productId=productId, category=category, limit=limit)

# New endpoint for advanced ML-based recommendations using User_Behavior and Viewed_Products
@app.get("/api/advanced-recommendations", response_model=Recommendation)
async def get_advanced_recommendations(
    customerId: Optional[int] = Query(None, description="Customer ID for logged-in users"),
    sessionId: Optional[str] = Query(None, description="Session ID for anonymous users"),
    category: Optional[str] = Query(None, description="Optional category filter"),
    limit: int = Query(10, description="Maximum number of recommendations to return")
):
    try:
        # Get advanced recommendations from advanced model using behavioral data
        recommendations = advanced_model.get_advanced_recommendations(
            customer_id=customerId, 
            session_id=sessionId,
            category=category, 
            limit=limit
        )
        
        if not recommendations:
            return {
                "success": True,
                "recommendations": [],
                "message": "No recommendations available for the given parameters"
            }
        
        return {
            "success": True,
            "recommendations": recommendations,
            "count": len(recommendations)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/api/recommendations/health", response_model=HealthResponse)
async def health_check():
    standard_model_status = "loaded" if standard_model.product_ids is not None else "not loaded"
    advanced_model_status = "loaded" if advanced_model.product_ids is not None else "not loaded" 
    product_count = len(standard_model.product_ids) if standard_model.product_ids is not None else 0
    
    return {
        "status": "healthy",
        "standard_model_status": standard_model_status,
        "advanced_model_status": advanced_model_status,
        "product_count": product_count
    }

if __name__ == '__main__':
    # Check if models directory exists, create if not
    if not os.path.exists('models'):
        os.makedirs('models')
    
    uvicorn.run(app, host="0.0.0.0", port=8003)
