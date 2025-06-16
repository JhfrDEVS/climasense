# pip install sqlitecloud flask flask-cors pandas statsmodels
import traceback
import time
import queue
import threading
import warnings
from functools import wraps
from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ValueWarning
import sqlitecloud
import logging

# Suprimir warning de statsmodels sobre frecuencia de fechas
warnings.filterwarnings('ignore', category=ValueWarning)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuración de la base de datos
DATABASE_URL = "sqlitecloud://camb7islnz.g2.sqlite.cloud:8860/chinook.sqlite?apikey=XEWJ9eYCYVMAFvtxhvojT9lmZyZecL4lg3WzHZVTbks"

app = Flask(__name__)
CORS(app)

class SQLiteCloudPool:
    """Pool de conexiones robusto para SQLite Cloud"""
    def __init__(self, connection_string, max_connections=3):
        self.connection_string = connection_string
        self.max_connections = max_connections
        self.pool = queue.Queue(maxsize=max_connections)
        self.lock = threading.Lock()
        self.connection_attempts = 0
        self.successful_connections = 0
        
    def get_connection(self):
        self.connection_attempts += 1
        try:
            # Intentar obtener conexión del pool
            conn = self.pool.get_nowait()
            # Verificar si la conexión está activa
            try:
                conn.execute("SELECT 1")
                self.successful_connections += 1
                return conn
            except:
                # Conexión muerta, cerrar y crear nueva
                try:
                    conn.close()
                except:
                    pass
                return self._create_new_connection()
        except queue.Empty:
            # No hay conexiones disponibles, crear nueva
            return self._create_new_connection()
    
    def _create_new_connection(self):
        conn = sqlitecloud.connect(self.connection_string)
        self.successful_connections += 1
        return conn
    
    def return_connection(self, conn):
        try:
            # Verificar si la conexión sigue activa antes de devolverla al pool
            conn.execute("SELECT 1")
            self.pool.put_nowait(conn)
        except queue.Full:
            # Pool lleno, cerrar conexión
            try:
                conn.close()
            except:
                pass
        except:
            # Conexión muerta, cerrarla
            try:
                conn.close()
            except:
                pass
    
    def get_stats(self):
        return {
            "connection_attempts": self.connection_attempts,
            "successful_connections": self.successful_connections,
            "success_rate": f"{(self.successful_connections/self.connection_attempts*100):.2f}%" if self.connection_attempts > 0 else "N/A"
        }

# Crear pool global
db_pool = SQLiteCloudPool(DATABASE_URL)

def retry_db_operation(max_retries=3, delay=1):
    """Decorador para reintentar operaciones de base de datos"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (sqlitecloud.exceptions.SQLiteCloudException, ConnectionResetError) as e:
                    last_exception = e
                    error_msg = str(e).lower()
                    
                    if any(keyword in error_msg for keyword in ['socket', 'connection', 'reset', 'timeout']):
                        if attempt < max_retries - 1:
                            wait_time = delay * (2 ** attempt)  # Backoff exponencial
                            logging.warning(f"Conexión perdida, reintentando en {wait_time}s... (intento {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                            continue
                    raise e
                except Exception as e:
                    logging.error(f"Error inesperado en DB: {str(e)}")
                    raise e
            
            raise last_exception
        return wrapper
    return decorator

@retry_db_operation(max_retries=3, delay=1)
def execute_query_safe(query, params=None):
    """Ejecutar consulta con manejo seguro de conexiones"""
    conn = None
    try:
        conn = db_pool.get_connection()
        cursor = conn.execute(query, params or [])
        
        # Obtener resultados según el tipo de consulta
        if query.strip().upper().startswith('SELECT'):
            result = cursor.fetchall()
        else:
            result = cursor.rowcount
            
        return result
        
    finally:
        if conn:
            db_pool.return_connection(conn)

@app.route("/predict_rain")
def predict_rain():
    try:
        query = """
            SELECT timestamp, dht11_humidity, bmp180_pressure 
            FROM sensor_readings 
            ORDER BY timestamp ASC
        """
        
        # Ejecutar consulta con reintentos automáticos
        rows = execute_query_safe(query)
        
        if not rows:
            return jsonify({
                "error": "No sensor data available",
                "will_rain": False,
                "prediction": []
            }), 404

        # Procesar datos
        df = pd.DataFrame(rows, columns=["timestamp", "humidity", "pressure"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        
        # Verificar que tenemos suficientes datos
        if len(df) < 10:
            return jsonify({
                "error": "Insufficient data for prediction",
                "will_rain": False,
                "prediction": []
            }), 400

        y = df["humidity"]
        exog = df[["pressure"]]
        
        # Especificar frecuencia para evitar warning de statsmodels
        if len(df) > 1:
            # Intentar inferir frecuencia
            try:
                df = df.asfreq(pd.infer_freq(df.index))
            except:
                # Si no puede inferir, usar frecuencia horaria por defecto
                df = df.asfreq('H')
                y = df["humidity"].dropna()
                exog = df[["pressure"]].dropna()

        # Crear y ajustar modelo ARIMA
        model = ARIMA(y, exog=exog, order=(2, 1, 2))
        model_fit = model.fit()

        # Realizar predicción
        future_exog = exog.tail(1).values.repeat(90, axis=0)
        forecast = model_fit.forecast(steps=90, exog=future_exog)
        prediction_list = forecast.tolist()

        # Determinar si lloverá (humedad > 85%)
        will_rain = any(h > 85 for h in prediction_list)
        
        # Estadísticas adicionales
        avg_humidity = sum(prediction_list) / len(prediction_list)
        max_humidity = max(prediction_list)
        min_humidity = min(prediction_list)

        return jsonify({
            "prediction": prediction_list,
            "will_rain": will_rain,
            "statistics": {
                "average_humidity": round(avg_humidity, 2),
                "max_humidity": round(max_humidity, 2),
                "min_humidity": round(min_humidity, 2),
                "data_points_used": len(df),
                "forecast_days": 90
            }
        })
        
    except sqlitecloud.exceptions.SQLiteCloudException as e:
        logging.error(f"Error de SQLite Cloud: {str(e)}")
        logging.error(f"Traceback completo: {traceback.format_exc()}")
        return jsonify({
            "error": "Database connection failed. Please try again.",
            "will_rain": False,
            "prediction": []
        }), 500
        
    except Exception as e:
        logging.error(f"Error completo: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "error": "An error occurred while processing the prediction.",
            "will_rain": False,
            "prediction": []
        }), 500

@app.route("/health")
def health_check():
    """Endpoint para verificar el estado de la aplicación"""
    try:
        # Probar conexión a la base de datos
        result = execute_query_safe("SELECT 1 as test")
        db_status = "connected" if result else "disconnected"
        
        stats = db_pool.get_stats()
        
        return jsonify({
            "status": "healthy",
            "database": db_status,
            "connection_stats": stats,
            "timestamp": pd.Timestamp.now().isoformat()
        })
    except Exception as e:
        logging.error(f"Error en health check: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        }), 500

@app.route("/sensor_stats")
def sensor_stats():
    """Obtener estadísticas básicas de los sensores"""
    try:
        query = """
            SELECT 
                COUNT(*) as total_readings,
                MIN(timestamp) as first_reading,
                MAX(timestamp) as last_reading,
                AVG(dht11_humidity) as avg_humidity,
                AVG(bmp180_pressure) as avg_pressure
            FROM sensor_readings
        """
        
        result = execute_query_safe(query)
        
        if result:
            row = result[0]
            return jsonify({
                "total_readings": row[0],
                "first_reading": row[1],
                "last_reading": row[2],
                "average_humidity": round(float(row[3]) if row[3] else 0, 2),
                "average_pressure": round(float(row[4]) if row[4] else 0, 2)
            })
        else:
            return jsonify({
                "error": "No sensor data found"
            }), 404
            
    except Exception as e:
        logging.error(f"Error obteniendo estadísticas: {str(e)}")
        return jsonify({
            "error": "Failed to retrieve sensor statistics"
        }), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Error no manejado: {str(e)}")
    logging.error(f"Traceback: {traceback.format_exc()}")
    return jsonify({
        "error": "An internal server error occurred."
    }), 500

if __name__ == "__main__":
    # Probar conexión inicial
    try:
        test_result = execute_query_safe("SELECT 1")
        logging.info("✅ Conexión inicial exitosa a SQLite Cloud")
        logging.info(f"✅ Resultado de prueba: {test_result}")
    except Exception as e:
        logging.error(f"❌ Error en conexión inicial: {str(e)}")
        logging.error("La aplicación seguirá ejecutándose con reintentos automáticos")
    
    logging.info("🚀 Iniciando servidor Flask para ClimateSense...")
    app.run(debug=True, host='0.0.0.0', port=5000)  