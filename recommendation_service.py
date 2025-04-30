import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import logging
import traceback
from typing import List, Dict, Any, Optional
import psycopg2
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('recommendation_service')

# Database connection function
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
        logger.error(f"Database connection error: {str(e)}")
        return None

class RecommendationModel:
    def __init__(self, model_type="standard", model_path=None):
        """
        Initialize recommendation model with specified type
        
        Args:
            model_type: Type of recommendation model ('standard' or 'advanced')
            model_path: Optional custom path for the model file
        """
        self.model_type = model_type
        
        # Set default model paths based on model type if not specified
        if model_path is None:
            if model_type == "standard":
                self.model_path = "models/standard_recommendation_model.pkl"
            else:
                self.model_path = "models/advanced_recommendation_model.pkl"
        else:
            self.model_path = model_path
            
        self.item_features = None
        self.product_ids = None
        self.similarity_matrix = None
        self.product_categories = {}  # Store categories for filtering
        
        # Load model if it exists
        if os.path.exists(self.model_path):
            try:
                self.load_model()
                logger.info(f"{model_type.capitalize()} model loaded successfully from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load {model_type} model: {str(e)}")
                logger.error(traceback.format_exc())
    
    def train(self, behavior_data: pd.DataFrame, product_data: pd.DataFrame):
        """
        Train recommendation model using user behavior and product data
        
        Args:
            behavior_data: DataFrame with columns (customer_id, session_id, product_id, event_type, created_at)
            product_data: DataFrame with columns (id, name, category, price, description)
        """
        try:
            logger.info(f"Training {self.model_type} model with {len(behavior_data)} behavior records and {len(product_data)} products")
            
            if product_data.empty:
                logger.error("Cannot train with empty product data")
                raise ValueError("Product data is empty")
                
            # Validate required columns in product_data
            required_columns = ['id', 'name', 'category']
            missing_columns = [col for col in required_columns if col not in product_data.columns]
            if missing_columns:
                msg = f"Missing required columns in product data: {missing_columns}"
                logger.error(msg)
                raise ValueError(msg)
            
            # Clean product data - handle missing values
            logger.info("Cleaning product data")
            product_data['name'] = product_data['name'].fillna("")
            product_data['category'] = product_data['category'].fillna("unknown")
            product_data['description'] = product_data['description'].fillna("")
            
            # Store category information for later filtering
            self.product_categories = dict(zip(product_data['id'], product_data['category']))
            
            # Process product data for content-based filtering
            # For standard model, we focus on content similarity
            if self.model_type == "standard":
                # Create text features combining name, category, and description with higher weight on name and category
                logger.info("Creating text features for standard model (content-focused)")
                product_data['features'] = product_data.apply(
                    lambda row: f"{row['name']} {row['name']} {row['category']} {row['category']} {row.get('description', '')}", 
                    axis=1
                )
            # For advanced model, we incorporate user behavior if available
            else:
                logger.info("Creating features for advanced model (behavior-aware)")
                # Still use product content but prepare for behavior enhancement
                product_data['features'] = product_data.apply(
                    lambda row: f"{row['name']} {row['category']} {row.get('description', '')}", 
                    axis=1
                )
                
                # Enhance features with behavior data if available
                if not behavior_data.empty:
                    logger.info(f"Incorporating {len(behavior_data)} behavior records into advanced model")
                    # This is where we could add more sophisticated behavior-based features
                    # For now, we'll use the basic content features for both models
            
            # Generate TF-IDF vectors for product features
            logger.info("Generating TF-IDF vectors")
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(product_data['features'])
        
            # Calculate cosine similarity between products
            logger.info("Calculating cosine similarity")
            self.similarity_matrix = cosine_similarity(tfidf_matrix)
            self.product_ids = product_data['id'].tolist()
        
            # Save the model
            self.save_model()
            logger.info(f"{self.model_type.capitalize()} model training completed with {len(self.product_ids)} products")
        
            return self
        except Exception as e:
            logger.error(f"Error during {self.model_type} model training: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def save_model(self):
        """Save trained model to disk"""
        try:
            logger.info(f"Saving {self.model_type} model to {self.model_path}")
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Verify we have data to save
            if self.similarity_matrix is None:
                raise ValueError("No similarity matrix to save")
            if self.product_ids is None or len(self.product_ids) == 0:
                raise ValueError("No product IDs to save")
                
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'similarity_matrix': self.similarity_matrix,
                        'product_ids': self.product_ids,
                        'product_categories': self.product_categories
                }, f)
                logger.info(f"{self.model_type.capitalize()} model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save {self.model_type} model: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def load_model(self):
        """Load trained model from disk"""
        try:
            logger.info(f"Loading {self.model_type} model from {self.model_path}")
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.similarity_matrix = data['similarity_matrix']
                self.product_ids = data['product_ids']
                self.product_categories = data.get('product_categories', {})
                logger.info(f"Loaded {self.model_type} model with {len(self.product_ids)} products")
        except Exception as e:
            logger.error(f"Error loading {self.model_type} model: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def get_recommendations(self, product_id: str, category: Optional[str] = None, limit: int = 4) -> List[str]:
        """
        Get content-based recommendations for a product
        
        Args:
            product_id: ID of the product to get recommendations for
            category: Optional category to filter recommendations
            limit: Maximum number of recommendations to return
            
        Returns:
            List of recommended product IDs
        """
        if self.similarity_matrix is None or self.product_ids is None:
            logger.warning(f"{self.model_type.capitalize()} model not trained or loaded - no recommendations available")
            return []
        
        try:
            # Find the index of the product in our data
            try:
                product_idx = self.product_ids.index(product_id)
            except ValueError:
                logger.warning(f"Product ID {product_id} not found in {self.model_type} model data")
                return []
            
            # Get similarity scores for this product with all others
            similarity_scores = list(enumerate(self.similarity_matrix[product_idx]))
            
            # Sort by similarity score (descending)
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            
            # Filter out the product itself
            similarity_scores = [s for s in similarity_scores if self.product_ids[s[0]] != product_id]
            
            # Get recommended product IDs
            recommendations = []
            product_indices = [i[0] for i in similarity_scores]
            
            # Apply category filter if specified
            if category:
                logger.info(f"Filtering recommendations by category: {category}")
                filtered_indices = []
                for idx in product_indices:
                    product_id_to_check = self.product_ids[idx]
                    if product_id_to_check in self.product_categories:
                        if self.product_categories[product_id_to_check] == category:
                            filtered_indices.append(idx)
                
                recommendations = [self.product_ids[i] for i in filtered_indices[:limit]]
            else:
                recommendations = [self.product_ids[i] for i in product_indices[:limit]]
            
            logger.info(f"Found {len(recommendations)} recommendations for product {product_id} using {self.model_type} model")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations with {self.model_type} model: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def get_advanced_recommendations(
        self,
        customer_id: Optional[int] = None,
        session_id: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[str]:
        """
        Simple advanced recommendation:
        For authenticated users, get 5 latest viewed products from Viewed_Products.
        For guests, get 5 latest from User_Behavior (event_type='product_viewed').
        For each, use get_recommendations to find similar products, combine, deduplicate, and return up to `limit`.
        """
        try:
            logger.info(f"[Simple Advanced] Getting recommendations for customer_id={customer_id}, session_id={session_id}, category={category}")
            conn = get_db_connection()
            if not conn:
                logger.error("Failed to connect to database for advanced recommendations")
                return self._fallback_to_popular_products(limit, category)
            try:
                # Step 1: Get up to 5 latest viewed products
                viewed_products = []
                if customer_id:
                    viewed_products = self._get_customer_viewed_products(conn, customer_id, 5)
                elif session_id:
                    # For guests, get from User_Behavior table
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT entity_id
                        FROM "User_Behavior"
                        WHERE session_id = %s AND event_type = 'product_viewed' AND entity_type = 'product'
                        ORDER BY created_at DESC
                        LIMIT 5
                    ''', (session_id,))
                    viewed_products = [row[0] for row in cursor.fetchall()]
                    cursor.close()
                else:
                    logger.warning("No customer_id or session_id provided for recommendations.")
                    return self._fallback_to_popular_products(limit, category)
                if not viewed_products:
                    logger.info("No viewed products found; fallback to popular.")
                    return self._fallback_to_popular_products(limit, category)
                # Step 2: For each viewed product, get similar products
                recommended = []
                seen = set(viewed_products)
                for pid in viewed_products:
                    sims = self.get_recommendations(pid, category=category, limit=limit)
                    for rec in sims:
                        if rec not in seen and rec not in recommended:
                            recommended.append(rec)
                        if len(recommended) >= limit:
                            break
                    if len(recommended) >= limit:
                        break
                logger.info(f"Returning {len(recommended)} advanced recommendations.")
                return recommended[:limit]
            finally:
                conn.close()
                logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error generating advanced recommendations: {str(e)}")
            logger.error(traceback.format_exc())
            return self._fallback_to_popular_products(limit, category)
            
    def _get_customer_viewed_products(self, conn, customer_id: int, limit: int) -> List[str]:
        """Get recently viewed products for a customer"""
        try:
            cursor = conn.cursor()
            query = """
            SELECT product_id FROM "Viewed_Products"
            WHERE customer_id = %s
            ORDER BY viewed_at DESC
            LIMIT %s
            """
            cursor.execute(query, (customer_id, limit))
            results = cursor.fetchall()
            cursor.close()
            
            return [row[0] for row in results]
        except Exception as e:
            logger.error(f"Error getting viewed products: {str(e)}")
            return []
            
    def _get_customer_behavior_products(self, conn, customer_id: int, limit: int) -> List[str]:
        """Get products from customer behavior data, including orders and payments"""
        try:
            cursor = conn.cursor()
            
            # Get products directly from user behavior
            behavior_query = """
            SELECT entity_id, 
                SUM(CASE 
                    WHEN event_type = 'product_added_to_cart' THEN 3
                    WHEN event_type = 'product_viewed' THEN 2
                    WHEN event_type = 'product_click' THEN 1
                    WHEN event_type = 'order_created' THEN 4
                    WHEN event_type = 'payment_completed' THEN 5
                    ELSE 1
                END) as weight
            FROM "User_Behavior"
            WHERE customer_id = %s 
            AND entity_type = 'product'
            AND created_at > NOW() - INTERVAL '30 days'
            GROUP BY entity_id
            """
            
            # Get products from order history
            order_query = """
            SELECT p.id as product_id, COUNT(*) * 6 as weight
            FROM "Orders" o
            JOIN "Order_Detail" od ON o.id = od.order_id
            JOIN "Products" p ON od.product_id = p.id
            WHERE o.customer_id = %s
            AND o.status IN ('completed', 'delivered', 'processing')
            AND o.created_at > NOW() - INTERVAL '60 days'
            GROUP BY p.id
            """
            
            # Get products associated with order events
            order_events_query = """
            SELECT p.id as product_id,
                SUM(CASE 
                    WHEN ub.event_type = 'order_created' THEN 4
                    WHEN ub.event_type = 'payment_completed' THEN 5
                    ELSE 0
                END) as weight
            FROM "User_Behavior" ub
            JOIN "Orders" o ON ub.entity_id = o.id::text
            JOIN "Order_Detail" od ON o.id = od.order_id
            JOIN "Products" p ON od.product_id = p.id
            WHERE ub.customer_id = %s
            AND ub.entity_type IN ('order', 'payment')
            AND ub.event_type IN ('order_created', 'payment_completed')
            AND ub.created_at > NOW() - INTERVAL '60 days'
            GROUP BY p.id
            """
            
            # Combine all queries with UNION ALL
            combined_query = f"""
            WITH behavior_products AS ({behavior_query}),
            order_products AS ({order_query}),
            order_event_products AS ({order_events_query})
            
            SELECT product_id, SUM(weight) as total_weight FROM (
                SELECT entity_id::uuid as product_id, weight FROM behavior_products
                UNION ALL
                SELECT product_id, weight FROM order_products
                UNION ALL
                SELECT product_id, weight FROM order_event_products
            ) combined
            GROUP BY product_id
            ORDER BY total_weight DESC
            LIMIT %s
            """
            
            cursor.execute(combined_query, (customer_id, customer_id, customer_id, limit))
            results = cursor.fetchall()
            cursor.close()
            
            return [row[0] for row in results]
        except Exception as e:
            logger.error(f"Error getting behavior products: {str(e)}")
            return []
            
    def _get_session_behavior_products(self, conn, session_id: str, limit: int) -> List[str]:
        """Get products from session behavior data, including orders and payments"""
        try:
            cursor = conn.cursor()
            
            # Get products directly from session behavior
            behavior_query = """
            SELECT entity_id,
                SUM(CASE 
                    WHEN event_type = 'product_added_to_cart' THEN 3
                    WHEN event_type = 'product_viewed' THEN 2
                    WHEN event_type = 'product_click' THEN 1
                    WHEN event_type = 'order_created' THEN 4
                    WHEN event_type = 'payment_completed' THEN 5
                    ELSE 1
                END) as weight
            FROM "User_Behavior"
            WHERE session_id::text = %s 
            AND entity_type = 'product'
            GROUP BY entity_id
            """
            
            # Get products from order events in this session
            order_events_query = """
            SELECT p.id as product_id,
                SUM(CASE 
                    WHEN ub.event_type = 'order_created' THEN 4
                    WHEN ub.event_type = 'payment_completed' THEN 5
                    ELSE 0
                END) as weight
            FROM "User_Behavior" ub
            JOIN "Orders" o ON ub.entity_id = o.id::text
            JOIN "Order_Detail" od ON o.id = od.order_id
            JOIN "Products" p ON od.product_id = p.id
            WHERE ub.session_id::text = %s
            AND ub.entity_type IN ('order', 'payment')
            AND ub.event_type IN ('order_created', 'payment_completed')
            GROUP BY p.id
            """
            
            # Combine all queries with UNION ALL
            combined_query = f"""
            WITH behavior_products AS ({behavior_query}),
            order_event_products AS ({order_events_query})
            
            SELECT product_id, SUM(weight) as total_weight FROM (
                SELECT entity_id::uuid as product_id, weight FROM behavior_products
                UNION ALL
                SELECT product_id, weight FROM order_event_products
            ) combined
            GROUP BY product_id
            ORDER BY total_weight DESC
            LIMIT %s
            """
            
            cursor.execute(combined_query, (session_id, session_id, limit))
            results = cursor.fetchall()
            cursor.close()
            
            return [row[0] for row in results]
        except Exception as e:
            logger.error(f"Error getting session behavior products: {str(e)}")
            return []
            
    def _get_preferred_categories(self, conn, customer_id: Optional[int], session_id: Optional[str]) -> List[str]:
        """Get user's preferred categories based on behavior, including orders and payments"""
        try:
            cursor = conn.cursor()
            
            if customer_id:
                # For logged-in users, use behavior, viewed products, and order history
                query = """
                WITH behavior_categories AS (
                    SELECT 
                        p.category,
                        SUM(CASE
                            WHEN ub.event_type = 'product_added_to_cart' THEN 3
                            WHEN ub.event_type = 'product_viewed' THEN 1
                            WHEN ub.event_type = 'product_click' THEN 2
                            WHEN ub.event_type = 'order_created' THEN 4
                            WHEN ub.event_type = 'payment_completed' THEN 5
                            ELSE 1
                        END) as weighted_count
                    FROM "User_Behavior" ub
                    JOIN "Products" p ON ub.entity_id::uuid = p.id
                    WHERE ub.customer_id = %s
                    AND ub.entity_type = 'product'
                    GROUP BY p.category
                ),
                viewed_categories AS (
                    SELECT p.category, COUNT(*) * 2 as weighted_count
                    FROM "Viewed_Products" vp
                    JOIN "Products" p ON vp.product_id = p.id
                    WHERE vp.customer_id = %s
                    GROUP BY p.category
                ),
                order_categories AS (
                    SELECT p.category, COUNT(*) * 6 as weighted_count
                    FROM "Orders" o
                    JOIN "Order_Detail" od ON o.id = od.order_id
                    JOIN "Products" p ON od.product_id = p.id
                    WHERE o.customer_id = %s
                    AND o.status IN ('completed', 'delivered', 'processing')
                    GROUP BY p.category
                ),
                order_events AS (
                    SELECT 
                        p.category, 
                        SUM(CASE
                            WHEN ub.event_type = 'order_created' THEN 4
                            WHEN ub.event_type = 'payment_completed' THEN 5
                            ELSE 0
                        END) as weighted_count
                    FROM "User_Behavior" ub
                    JOIN "Orders" o ON ub.entity_id = o.id::text
                    JOIN "Order_Detail" od ON o.id = od.order_id
                    JOIN "Products" p ON od.product_id = p.id
                    WHERE ub.customer_id = %s
                    AND ub.entity_type IN ('order', 'payment')
                    AND ub.event_type IN ('order_created', 'payment_completed')
                    GROUP BY p.category
                ),
                combined_categories AS (
                    SELECT category, weighted_count FROM behavior_categories
                    UNION ALL
                    SELECT category, weighted_count FROM viewed_categories
                    UNION ALL
                    SELECT category, weighted_count FROM order_categories
                    UNION ALL
                    SELECT category, weighted_count FROM order_events
                )
                SELECT category, SUM(weighted_count) as total_weight
                FROM combined_categories
                GROUP BY category
                ORDER BY total_weight DESC
                LIMIT 5
                """
                cursor.execute(query, (customer_id, customer_id, customer_id, customer_id))
            elif session_id:
                # For anonymous users, use session behavior and any order events
                query = """
                WITH behavior_categories AS (
                    SELECT 
                        p.category,
                        SUM(CASE
                            WHEN ub.event_type = 'product_added_to_cart' THEN 3
                            WHEN ub.event_type = 'product_viewed' THEN 1
                            WHEN ub.event_type = 'product_click' THEN 2
                            WHEN ub.event_type = 'order_created' THEN 4
                            WHEN ub.event_type = 'payment_completed' THEN 5
                            ELSE 1
                        END) as weighted_count
                    FROM "User_Behavior" ub
                    JOIN "Products" p ON ub.entity_id::uuid = p.id
                    WHERE ub.session_id::text = %s
                    AND ub.entity_type = 'product'
                    GROUP BY p.category
                ),
                order_events AS (
                    SELECT 
                        p.category, 
                        SUM(CASE
                            WHEN ub.event_type = 'order_created' THEN 4
                            WHEN ub.event_type = 'payment_completed' THEN 5
                            ELSE 0
                        END) as weighted_count
                    FROM "User_Behavior" ub
                    JOIN "Orders" o ON ub.entity_id = o.id::text
                    JOIN "Order_Detail" od ON o.id = od.order_id
                    JOIN "Products" p ON od.product_id = p.id
                    WHERE ub.session_id::text = %s
                    AND ub.entity_type IN ('order', 'payment')
                    AND ub.event_type IN ('order_created', 'payment_completed')
                    GROUP BY p.category
                ),
                combined_categories AS (
                    SELECT category, weighted_count FROM behavior_categories
                    UNION ALL
                    SELECT category, weighted_count FROM order_events
                )
                SELECT category, SUM(weighted_count) as total_weight
                FROM combined_categories
                GROUP BY category
                ORDER BY total_weight DESC
                LIMIT 5
                """
                cursor.execute(query, (session_id, session_id))
            else:
                # Fallback to popular categories
                query = """
                SELECT category, COUNT(*) as count
                FROM "Products"
                WHERE status = 'active'
                GROUP BY category
                ORDER BY count DESC
                LIMIT 5
                """
                cursor.execute(query)
                
            results = cursor.fetchall()
            cursor.close()
            
            return [row[0] for row in results]
        except Exception as e:
            logger.error(f"Error getting preferred categories: {str(e)}")
            return []
            
    def _get_top_products_by_category(self, conn, category: str, limit: int) -> List[str]:
        """Get top products in a category"""
        try:
            cursor = conn.cursor()
            
            query = """
            SELECT p.id
            FROM "Products" p
            LEFT JOIN "Rating_Comment" rc ON p.id = rc.product_id
            WHERE p.category = %s
            AND p.status = 'active'
            GROUP BY p.id
            ORDER BY COUNT(rc.id) DESC, RANDOM()
            LIMIT %s
            """
            
            cursor.execute(query, (category, limit))
            results = cursor.fetchall()
            cursor.close()
            
            return [row[0] for row in results]
        except Exception as e:
            logger.error(f"Error getting category products: {str(e)}")
            return []
            
    def _get_popular_products(self, conn, limit: int, exclude_ids: List[str] = None) -> List[str]:
        """Get generally popular products, considering order and payment events"""
        try:
            cursor = conn.cursor()
            
            if exclude_ids and len(exclude_ids) > 0:
                placeholders = ','.join(['%s' for _ in exclude_ids])
                query = f"""
                WITH behavior_counts AS (
                    SELECT 
                        entity_id, 
                        SUM(CASE
                            WHEN event_type = 'product_added_to_cart' THEN 3
                            WHEN event_type = 'product_viewed' THEN 1
                            WHEN event_type = 'product_click' THEN 2
                            ELSE 1
                        END) as weighted_count
                    FROM "User_Behavior"
                    WHERE entity_type = 'product'
                    AND entity_id NOT IN ({placeholders})
                    GROUP BY entity_id
                ),
                order_products AS (
                    SELECT 
                        p.id as product_id, 
                        COUNT(*) * 4 as weighted_count
                    FROM "Orders" o
                    JOIN "Order_Detail" od ON o.id = od.order_id
                    JOIN "Products" p ON od.product_id = p.id
                    WHERE p.id NOT IN ({placeholders})
                    AND o.status IN ('completed', 'delivered', 'processing')
                    GROUP BY p.id
                ),
                order_events AS (
                    SELECT 
                        p.id as product_id,
                        SUM(CASE 
                            WHEN ub.event_type = 'order_created' THEN 4
                            WHEN ub.event_type = 'payment_completed' THEN 5
                            ELSE 0 
                        END) as weighted_count
                    FROM "User_Behavior" ub
                    JOIN "Orders" o ON ub.entity_id = o.id::text
                    JOIN "Order_Detail" od ON o.id = od.order_id
                    JOIN "Products" p ON od.product_id = p.id
                    WHERE p.id NOT IN ({placeholders})
                    AND ub.entity_type IN ('order', 'payment')
                    AND ub.event_type IN ('order_created', 'payment_completed')
                    GROUP BY p.id
                ),
                combined_products AS (
                    SELECT entity_id::uuid as product_id, weighted_count FROM behavior_counts
                    UNION ALL
                    SELECT product_id, weighted_count FROM order_products
                    UNION ALL
                    SELECT product_id, weighted_count FROM order_events
                )
                SELECT product_id, SUM(weighted_count) as total_weight
                FROM combined_products
                GROUP BY product_id
                ORDER BY total_weight DESC
                LIMIT %s
                """
                cursor.execute(query, exclude_ids + exclude_ids + exclude_ids + [limit])
            else:
                query = """
                WITH behavior_counts AS (
                    SELECT 
                        entity_id, 
                        SUM(CASE
                            WHEN event_type = 'product_added_to_cart' THEN 3
                            WHEN event_type = 'product_viewed' THEN 1
                            WHEN event_type = 'product_click' THEN 2
                            ELSE 1
                        END) as weighted_count
                    FROM "User_Behavior"
                    WHERE entity_type = 'product'
                    GROUP BY entity_id
                ),
                order_products AS (
                    SELECT 
                        p.id as product_id, 
                        COUNT(*) * 4 as weighted_count
                    FROM "Orders" o
                    JOIN "Order_Detail" od ON o.id = od.order_id
                    JOIN "Products" p ON od.product_id = p.id
                    WHERE o.status IN ('completed', 'delivered', 'processing')
                    GROUP BY p.id
                ),
                order_events AS (
                    SELECT 
                        p.id as product_id,
                        SUM(CASE 
                            WHEN ub.event_type = 'order_created' THEN 4
                            WHEN ub.event_type = 'payment_completed' THEN 5
                            ELSE 0 
                        END) as weighted_count
                    FROM "User_Behavior" ub
                    JOIN "Orders" o ON ub.entity_id = o.id::text
                    JOIN "Order_Detail" od ON o.id = od.order_id
                    JOIN "Products" p ON od.product_id = p.id
                    WHERE ub.entity_type IN ('order', 'payment')
                    AND ub.event_type IN ('order_created', 'payment_completed')
                    GROUP BY p.id
                ),
                combined_products AS (
                    SELECT entity_id::uuid as product_id, weighted_count FROM behavior_counts
                    UNION ALL
                    SELECT product_id, weighted_count FROM order_products
                    UNION ALL
                    SELECT product_id, weighted_count FROM order_events
                )
                SELECT product_id, SUM(weighted_count) as total_weight
                FROM combined_products
                GROUP BY product_id
                ORDER BY total_weight DESC
                LIMIT %s
                """
                cursor.execute(query, (limit,))
                
            results = cursor.fetchall()
            cursor.close()
            
            return [row[0] for row in results]
        except Exception as e:
            logger.error(f"Error getting popular products: {str(e)}")
            return []
            
    def _fallback_to_popular_products(self, limit: int, category: Optional[str] = None) -> List[str]:
        """Fallback when database queries fail"""
        if self.product_ids is None:
            return []
            
        import random
        
        # Return random products, optionally filtered by category
        if category and self.product_categories:
            category_products = [pid for pid, cat in self.product_categories.items() 
                                if cat == category]
            if category_products:
                return random.sample(
                    category_products, 
                    min(limit, len(category_products))
                )
                
        # Otherwise return random products
        return random.sample(
            self.product_ids,
            min(limit, len(self.product_ids))
        )