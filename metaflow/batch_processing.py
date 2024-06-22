import os
from metaflow import FlowSpec, step, S3, Parameter
import data_preprocess

# Configuración de las credenciales de acceso a AWS S3 (minio)
os.environ['AWS_ACCESS_KEY_ID'] = "minio"
os.environ['AWS_SECRET_ACCESS_KEY'] = "minio123"
os.environ['AWS_ENDPOINT_URL_S3'] = "http://localhost:9000"

# pylint: disable=all

class BatchProcessingModel(FlowSpec):

    data_file_name = Parameter('data_file_name', default='data_playlist_batch.csv')

    @step
    def start(self):
        """
        Step para iniciar el flujo. Imprime un mensaje de inicio y avanza.
        """
        print("Starting Batch Prediction")
        self.next(self.load_data, self.load_model)

    @step
    def load_data(self):
        """
        Paso para cargar los datos de entrada de S3
        """
        import pandas as pd

        # Se utiliza el objeto S3 para acceder a los datos desde el bucket en S3.
        s3 = S3(s3root="s3://batch/")
        data_obj = s3.get(f"data/{self.data_file_name}")
        print(f"Loaded file: {self.data_file_name}")
        
        # Lee y procesa el dataframe de pandas:
        self.X_batch = pd.read_csv(data_obj.path)

        self.X_batch =  data_preprocess.transform(self.X_batch)
        self.next(self.batch_processing)

    @step
    def load_model(self):
        """
        Paso para cargar el modelo previamente entrenado.
        """
        # Se utiliza el objeto S3 para acceder al modelo desde el bucket en S3.
        import pickle
        s3 = S3(s3root="s3://batch/")
        model_param = s3.get("artifact/svm_model.pkl")

        # Levantamos una instancia del modelo entrenado sin necesidad de hacer fit.
        with open(model_param.path, 'rb') as file:
            loaded_model = pickle.load(file)

        # Mostramos el modelo cargado.
        print("THIS IS THE MODEL: ", loaded_model)

        self.model = loaded_model
        self.next(self.batch_processing)

    @step
    def batch_processing(self, previous_tasks):
        """
        Paso para realizar el procesamiento por lotes y obtener predicciones.
        """
        import numpy as np
        import hashlib

        print("Obtaining predictions")

        # Se recorren las tareas previas para obtener los datos y el modelo.
        for task in previous_tasks:
            if hasattr(task, 'X_batch'):
                data = task.X_batch
            if hasattr(task, 'model'):
                model = task.model

        # Se obtienen las predicciones utilizando el modelo.
        out = model.predict(data)

        # Se define un diccionario de mapeo
        label_map = {0: "NO_PLAYLIST", 1: "PLAYLIST"}
        labels = np.array([label_map[idx] for idx in out])

        # Se genera un hash para cada fila de datos.
        data['key'] = data.apply(lambda row: ' '.join(map(str, row)), axis=1)
        data['hashed'] = data['key'].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())

        # Preparamos los datos para ser enviados a Redis
        dict_redis = {}
        for index, row in data.iterrows():
            dict_redis[row["hashed"]] = labels[index]

        #Creamos el nuevo diccionario
        self.redis_data = dict_redis

        #Hacemos la ingesta para pasarle el dato a las redis
        self.next(self.ingest_redis)

    @step
    def ingest_redis(self):
        """
        Paso para ingestar los resultados en Redis.
        """
        import redis

        print("Ingesting Redis")
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)

        # Comenzamos un pipeline de Redis
        pipeline = r.pipeline()

        # Se pre-ingresan los datos en Redis.
        for key, value in self.redis_data.items():
            pipeline.set(key, value)

        # Ahora ingestamos todos de una y dejamos que Redis resuelva de la forma más eficiente
        pipeline.execute()

        # Le damos un aviso de que terminó
        self.next(self.end)

    @step
    def end(self):
        """
        Paso final del flujo. Imprime un mensaje de finalización.
        """
        print("Finished")


if __name__ == "__main__":
    BatchProcessingModel()
