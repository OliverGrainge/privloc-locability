import sqlite3
from typing import Optional, List, Dict, Any, Union
import numpy as np


class PredictionDatabase:
    """A class to manage the location prediction database."""
    
    def __init__(self, db_name='location_db.db'):
        """Initialize the database connection."""
        self.db_name = db_name
        self.setup()
    
    def get_connection(self):
        """Get a connection to the database."""
        return sqlite3.connect(self.db_name)
    
    def setup(self):
        """Create all tables if they don't exist."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                image_id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                image_name TEXT NOT NULL,
                true_latitude REAL,
                true_longitude REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
                UNIQUE(dataset_id, image_name)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                predicted_latitude REAL,
                predicted_longitude REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES images(image_id),
                UNIQUE(image_id, model_name)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    # ============ DATASET METHODS ============
    
    def add_dataset(self, dataset_name: str) -> bool:
        """
        Add a new dataset.
        
        Returns:
            bool: True if added successfully, False if already exists.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO datasets (dataset_name) VALUES (?)', (dataset_name,))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
    
    def get_datasets(self) -> List[str]:
        """Get all dataset names."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT dataset_name FROM datasets ORDER BY dataset_name')
        results = [row[0] for row in cursor.fetchall()]
        conn.close()
        return results
    
    # ============ IMAGE METHODS ============
    
    def add_image(self, dataset_name: str, image_name: str, 
                  true_latitude: float, true_longitude: float) -> bool:
        """
        Add a new image to a dataset.
        
        Returns:
            bool: True if added successfully, False otherwise.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO images (dataset_id, image_name, true_latitude, true_longitude)
                SELECT dataset_id, ?, ?, ?
                FROM datasets
                WHERE dataset_name = ?
            ''', (image_name, true_latitude, true_longitude, dataset_name))
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
    
    def add_images_batch(self, dataset_name: str, 
                        image_names: Union[List[str], np.ndarray],
                        latitudes: Union[List[float], np.ndarray],
                        longitudes: Union[List[float], np.ndarray]) -> Dict[str, int]:
        """
        Add multiple images to a dataset in a single transaction.
        
        Args:
            dataset_name: Name of the dataset
            image_names: List or numpy array of image names
            latitudes: List or numpy array of latitudes
            longitudes: List or numpy array of longitudes
        
        Returns:
            Dict with 'added' and 'skipped' counts
        """
        # Convert numpy arrays to lists if needed
        if isinstance(image_names, np.ndarray):
            image_names = image_names.tolist()
        if isinstance(latitudes, np.ndarray):
            latitudes = latitudes.tolist()
        if isinstance(longitudes, np.ndarray):
            longitudes = longitudes.tolist()
        
        if not (len(image_names) == len(latitudes) == len(longitudes)):
            raise ValueError("image_names, latitudes, and longitudes must have the same length")
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        added = 0
        skipped = 0
        
        try:
            for img_name, lat, lon in zip(image_names, latitudes, longitudes):
                try:
                    cursor.execute('''
                        INSERT INTO images (dataset_id, image_name, true_latitude, true_longitude)
                        SELECT dataset_id, ?, ?, ?
                        FROM datasets
                        WHERE dataset_name = ?
                    ''', (img_name, float(lat), float(lon), dataset_name))
                    if cursor.rowcount > 0:
                        added += 1
                    else:
                        skipped += 1
                except sqlite3.IntegrityError:
                    skipped += 1
            
            conn.commit()
        finally:
            conn.close()
        
        return {'added': added, 'skipped': skipped}
    
    def get_images(self, dataset_name: str) -> List[str]:
        """Get all image names in a dataset."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT i.image_name
            FROM images i
            JOIN datasets d ON i.dataset_id = d.dataset_id
            WHERE d.dataset_name = ?
            ORDER BY i.image_name
        ''', (dataset_name,))
        results = [row[0] for row in cursor.fetchall()]
        conn.close()
        return results
    
    # ============ PREDICTION METHODS ============
    
    def add_prediction(self, dataset_name: str, image_name: str, 
                      model_name: str, predicted_latitude: float, 
                      predicted_longitude: float) -> bool:
        """
        Add a prediction for an image from a model.
        
        Returns:
            bool: True if added successfully, False otherwise.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO predictions (image_id, model_name, predicted_latitude, predicted_longitude)
                SELECT i.image_id, ?, ?, ?
                FROM images i
                JOIN datasets d ON i.dataset_id = d.dataset_id
                WHERE d.dataset_name = ? AND i.image_name = ?
            ''', (model_name, predicted_latitude, predicted_longitude, dataset_name, image_name))
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
    
    def add_predictions_batch(self, dataset_name: str, model_name: str,
                             image_names: Union[List[str], np.ndarray],
                             predicted_latitudes: Union[List[float], np.ndarray],
                             predicted_longitudes: Union[List[float], np.ndarray]) -> Dict[str, int]:
        """
        Add multiple predictions from a single model in a single transaction.
        
        Args:
            dataset_name: Name of the dataset
            model_name: Name of the model
            image_names: List or numpy array of image names
            predicted_latitudes: List or numpy array of predicted latitudes
            predicted_longitudes: List or numpy array of predicted longitudes
        
        Returns:
            Dict with 'added' and 'skipped' counts
        """
        # Convert numpy arrays to lists if needed
        if isinstance(image_names, np.ndarray):
            image_names = image_names.tolist()
        if isinstance(predicted_latitudes, np.ndarray):
            predicted_latitudes = predicted_latitudes.tolist()
        if isinstance(predicted_longitudes, np.ndarray):
            predicted_longitudes = predicted_longitudes.tolist()
        
        if not (len(image_names) == len(predicted_latitudes) == len(predicted_longitudes)):
            raise ValueError("image_names, predicted_latitudes, and predicted_longitudes must have the same length")
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        added = 0
        skipped = 0
        
        try:
            for img_name, pred_lat, pred_lon in zip(image_names, predicted_latitudes, predicted_longitudes):
                try:
                    cursor.execute('''
                        INSERT INTO predictions (image_id, model_name, predicted_latitude, predicted_longitude)
                        SELECT i.image_id, ?, ?, ?
                        FROM images i
                        JOIN datasets d ON i.dataset_id = d.dataset_id
                        WHERE d.dataset_name = ? AND i.image_name = ?
                    ''', (model_name, float(pred_lat), float(pred_lon), dataset_name, img_name))
                    if cursor.rowcount > 0:
                        added += 1
                    else:
                        skipped += 1
                except sqlite3.IntegrityError:
                    skipped += 1
            
            conn.commit()
        finally:
            conn.close()
        
        return {'added': added, 'skipped': skipped}
    
    def update_prediction(self, dataset_name: str, image_name: str,
                         model_name: str, predicted_latitude: float,
                         predicted_longitude: float) -> bool:
        """
        Update an existing prediction.
        
        Returns:
            bool: True if updated successfully, False if prediction doesn't exist.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE predictions
            SET predicted_latitude = ?,
                predicted_longitude = ?
            WHERE image_id = (
                SELECT i.image_id
                FROM images i
                JOIN datasets d ON i.dataset_id = d.dataset_id
                WHERE d.dataset_name = ? AND i.image_name = ?
            ) AND model_name = ?
        ''', (predicted_latitude, predicted_longitude, 
              dataset_name, image_name, model_name))
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success
    
    def get_models(self, dataset_name: Optional[str] = None, 
                   image_name: Optional[str] = None) -> List[str]:
        """
        Get all model names, optionally filtered by dataset and/or image.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if dataset_name and image_name:
            cursor.execute('''
                SELECT DISTINCT p.model_name
                FROM predictions p
                JOIN images i ON p.image_id = i.image_id
                JOIN datasets d ON i.dataset_id = d.dataset_id
                WHERE d.dataset_name = ? AND i.image_name = ?
                ORDER BY p.model_name
            ''', (dataset_name, image_name))
        elif dataset_name:
            cursor.execute('''
                SELECT DISTINCT p.model_name
                FROM predictions p
                JOIN images i ON p.image_id = i.image_id
                JOIN datasets d ON i.dataset_id = d.dataset_id
                WHERE d.dataset_name = ?
                ORDER BY p.model_name
            ''', (dataset_name,))
        else:
            cursor.execute('SELECT DISTINCT model_name FROM predictions ORDER BY model_name')
        
        results = [row[0] for row in cursor.fetchall()]
        conn.close()
        return results
    
    # ============ QUERY METHODS ============
    
    def query(self, dataset_name: str, image_name: str, 
              model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Query predictions for an image.
        
        Args:
            dataset_name: Name of the dataset
            image_name: Name of the image
            model_name: Optional model name. If provided, returns only that model's prediction.
                       If None, returns all predictions for the image.
        
        Returns:
            Dictionary with structure:
            {
                'image': image_name,
                'dataset': dataset_name,
                'true_latitude': float,
                'true_longitude': float,
                'predictions': [
                    {
                        'model': model_name,
                        'predicted_latitude': float,
                        'predicted_longitude': float
                    },
                    ...
                ]
            }
            Returns None if image not found.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get image true location
        cursor.execute('''
            SELECT i.true_latitude, i.true_longitude
            FROM images i
            JOIN datasets d ON i.dataset_id = d.dataset_id
            WHERE d.dataset_name = ? AND i.image_name = ?
        ''', (dataset_name, image_name))
        
        image_result = cursor.fetchone()
        if not image_result:
            conn.close()
            return None
        
        true_lat, true_lon = image_result
        
        # Get predictions
        if model_name:
            cursor.execute('''
                SELECT p.model_name, p.predicted_latitude, p.predicted_longitude
                FROM predictions p
                JOIN images i ON p.image_id = i.image_id
                JOIN datasets d ON i.dataset_id = d.dataset_id
                WHERE d.dataset_name = ? AND i.image_name = ? AND p.model_name = ?
            ''', (dataset_name, image_name, model_name))
        else:
            cursor.execute('''
                SELECT p.model_name, p.predicted_latitude, p.predicted_longitude
                FROM predictions p
                JOIN images i ON p.image_id = i.image_id
                JOIN datasets d ON i.dataset_id = d.dataset_id
                WHERE d.dataset_name = ? AND i.image_name = ?
                ORDER BY p.model_name
            ''', (dataset_name, image_name))
        
        predictions = []
        for row in cursor.fetchall():
            predictions.append({
                'model': row[0],
                'predicted_latitude': row[1],
                'predicted_longitude': row[2]
            })
        
        conn.close()
        
        return {
            'image': image_name,
            'dataset': dataset_name,
            'true_latitude': true_lat,
            'true_longitude': true_lon,
            'predictions': predictions
        }
    
    def query_batch(self, dataset_name: str, image_names: Optional[List[str]] = None,
                   model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Query predictions for multiple images.
        
        Args:
            dataset_name: Name of the dataset
            image_names: List of image names. If None, queries all images in dataset.
            model_name: Optional model name to filter predictions.
        
        Returns:
            List of dictionaries, same structure as query() method.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get all images with their true locations
        if image_names:
            placeholders = ','.join('?' * len(image_names))
            cursor.execute(f'''
                SELECT i.image_name, i.true_latitude, i.true_longitude, i.image_id
                FROM images i
                JOIN datasets d ON i.dataset_id = d.dataset_id
                WHERE d.dataset_name = ? AND i.image_name IN ({placeholders})
                ORDER BY i.image_name
            ''', [dataset_name] + image_names)
        else:
            cursor.execute('''
                SELECT i.image_name, i.true_latitude, i.true_longitude, i.image_id
                FROM images i
                JOIN datasets d ON i.dataset_id = d.dataset_id
                WHERE d.dataset_name = ?
                ORDER BY i.image_name
            ''', (dataset_name,))
        
        images = cursor.fetchall()
        
        if not images:
            conn.close()
            return []
        
        # Get all predictions for these images
        image_ids = [img[3] for img in images]
        placeholders = ','.join('?' * len(image_ids))
        
        if model_name:
            cursor.execute(f'''
                SELECT p.image_id, p.model_name, p.predicted_latitude, p.predicted_longitude
                FROM predictions p
                WHERE p.image_id IN ({placeholders}) AND p.model_name = ?
                ORDER BY p.image_id, p.model_name
            ''', image_ids + [model_name])
        else:
            cursor.execute(f'''
                SELECT p.image_id, p.model_name, p.predicted_latitude, p.predicted_longitude
                FROM predictions p
                WHERE p.image_id IN ({placeholders})
                ORDER BY p.image_id, p.model_name
            ''', image_ids)
        
        all_predictions = cursor.fetchall()
        conn.close()
        
        # Organize predictions by image_id
        predictions_by_image = {}
        for pred in all_predictions:
            img_id = pred[0]
            if img_id not in predictions_by_image:
                predictions_by_image[img_id] = []
            predictions_by_image[img_id].append({
                'model': pred[1],
                'predicted_latitude': pred[2],
                'predicted_longitude': pred[3]
            })
        
        # Build result list
        results = []
        for img in images:
            img_name, true_lat, true_lon, img_id = img
            results.append({
                'image': img_name,
                'dataset': dataset_name,
                'true_latitude': true_lat,
                'true_longitude': true_lon,
                'predictions': predictions_by_image.get(img_id, [])
            })
        
        return results


# ============ EXAMPLE USAGE ============

if __name__ == '__main__':
    import numpy as np
    
    # Create a database instance
    db = LocationDatabase()
    
    # Add a dataset
    if db.add_dataset('im2gps'):
        print("Dataset 'im2gps' added successfully!")
    
    # ========== BATCH ADD IMAGES WITH NUMPY ARRAYS ==========
    print("\n=== Batch adding images with numpy arrays ===")
    image_names = np.array(['photo1.jpg', 'photo2.jpg', 'photo3.jpg', 'photo4.jpg'])
    latitudes = np.array([51.5074, 52.5200, 48.8566, 40.7128])
    longitudes = np.array([-0.1278, 13.4050, 2.3522, -74.0060])
    
    result = db.add_images_batch('im2gps', image_names, latitudes, longitudes)
    print(f"Added {result['added']} images, skipped {result['skipped']}")
    
    # ========== BATCH ADD PREDICTIONS WITH NUMPY ARRAYS ==========
    print("\n=== Batch adding predictions for GeoCLIP ===")
    predicted_lats = np.array([51.5080, 52.5190, 48.8570, 40.7130])
    predicted_lons = np.array([-0.1275, 13.4045, 2.3520, -74.0055])
    
    result = db.add_predictions_batch('im2gps', 'GeoCLIP', image_names, predicted_lats, predicted_lons)
    print(f"Added {result['added']} predictions, skipped {result['skipped']}")
    
    print("\n=== Batch adding predictions for ResNet ===")
    predicted_lats = np.array([51.5070, 52.5210, 48.8560, 40.7125])
    predicted_lons = np.array([-0.1280, 13.4055, 2.3525, -74.0065])
    
    result = db.add_predictions_batch('im2gps', 'ResNet', image_names, predicted_lats, predicted_lons)
    print(f"Added {result['added']} predictions, skipped {result['skipped']}")
    
    # ========== SINGLE IMAGE QUERY ==========
    print("\n=== Single image query (all models) ===")
    result = db.query('im2gps', 'photo1.jpg')
    if result:
        print(f"Image: {result['image']}")
        print(f"True location: ({result['true_latitude']}, {result['true_longitude']})")
        print(f"Predictions ({len(result['predictions'])} models):")
        for pred in result['predictions']:
            print(f"  {pred['model']}: ({pred['predicted_latitude']}, {pred['predicted_longitude']})")
    
    # ========== SINGLE IMAGE QUERY WITH SPECIFIC MODEL ==========
    print("\n=== Single image query (specific model) ===")
    result = db.query('im2gps', 'photo1.jpg', 'GeoCLIP')
    if result and result['predictions']:
        pred = result['predictions'][0]
        print(f"Image: {result['image']}")
        print(f"True: ({result['true_latitude']}, {result['true_longitude']})")
        print(f"GeoCLIP: ({pred['predicted_latitude']}, {pred['predicted_longitude']})")
    
    # ========== BATCH QUERY (SPECIFIC IMAGES) ==========
    print("\n=== Batch query (specific images) ===")
    results = db.query_batch('im2gps', ['photo1.jpg', 'photo2.jpg'])
    for result in results:
        print(f"\n{result['image']}:")
        print(f"  True: ({result['true_latitude']}, {result['true_longitude']})")
        print(f"  Predictions: {len(result['predictions'])} models")
        for pred in result['predictions']:
            print(f"    {pred['model']}: ({pred['predicted_latitude']}, {pred['predicted_longitude']})")
    
    # ========== BATCH QUERY (ALL IMAGES, SPECIFIC MODEL) ==========
    print("\n=== Batch query (all images, GeoCLIP only) ===")
    results = db.query_batch('im2gps', model_name='GeoCLIP')
    for result in results:
        print(f"{result['image']}: ", end='')
        if result['predictions']:
            pred = result['predictions'][0]
            print(f"({pred['predicted_latitude']}, {pred['predicted_longitude']})")
        else:
            print("No prediction")