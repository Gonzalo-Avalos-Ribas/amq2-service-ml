from metaflow import S3
HIGH_LINEAR_CORRELATION_VARIABLES= ['danceability','energy','speechiness']

def transform(input):
    import pickle
    input = input[HIGH_LINEAR_CORRELATION_VARIABLES]
    s3 = S3(s3root="s3://batch/")

    # Levantamos los transformers de cada variable. Los mismos fueron pre entrenados.
    transformer = s3.get("artifact/energy.pkl")
    with open(transformer.path, 'rb') as file:
        energy_transformer = pickle.load(file)
    
    transformer = s3.get("artifact/danceability.pkl")
    with open(transformer.path, 'rb') as file:
        danceability_transformer = pickle.load(file)

    transformer = s3.get("artifact/speechiness.pkl")
    with open(transformer.path, 'rb') as file:
        speechiness_transformer = pickle.load(file)

    transformers = {
        'energy': energy_transformer,
        'danceability':danceability_transformer,
        'speechiness': speechiness_transformer,
    }

    for _, col in enumerate(input.columns):
        input[col] =  transformers[col].transform(input[col].to_numpy().reshape((-1, 1)))

    return input