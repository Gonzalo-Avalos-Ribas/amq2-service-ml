HIGH_LINEAR_CORRELATION_VARIABLES= ['danceability','energy','speechiness']

def transform(input):
    import pickle
    input = input[HIGH_LINEAR_CORRELATION_VARIABLES]

    # Levantamos los transformers de cada variable. Los mismos fueron pre entrenados.
    with open("energy.pkl", 'rb') as file:
        energy_transformer = pickle.load(file)
    
    with open("danceability.pkl", 'rb') as file:
        danceability_transformer = pickle.load(file)

    with open("speechiness.pkl", 'rb') as file:
        speechiness_transformer = pickle.load(file)

    transformers = {
        'energy': energy_transformer,
        'danceability':danceability_transformer,
        'speechiness': speechiness_transformer,
    }

    for _, col in enumerate(input.columns):
        input[col] =  transformers[col].transform(input[col].to_numpy().reshape((-1, 1)))

    return input