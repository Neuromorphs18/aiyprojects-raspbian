import numpy as np
import keras


def normalize_features(features, order=None):
    return features / np.linalg.norm(features, order, 1, True)


def get_embeddings_from_labels(labels, w2v_embeddings):
    embeddings = np.zeros((len(labels), w2v_embeddings.shape[1]))

    for i, label in enumerate(labels):
        embeddings[i] = w2v_embeddings[label]

    return embeddings


def get_labels_from_embeddings(predicted_embeddings, w2v_embeddings):

    proximity_scores = np.dot(predicted_embeddings, w2v_embeddings.transpose())

    return np.argmax(proximity_scores, -1)


path_to_embeddings = './data/cifar10_w2v_embeddings.npz'
path_to_cifar_fts = './data/output.npz'

# load w2v vectors and labels
word_features = np.load(path_to_embeddings)['embeddings']
word_features = normalize_features(word_features)
word_labels = np.load(path_to_embeddings)['words']

# load visual vectors and labels
vision_features = np.load(path_to_cifar_fts)['features']
vision_features = normalize_features(vision_features)
vision_labels_onehot = np.load(path_to_cifar_fts)['labels']
vision_labels = np.argmax(vision_labels_onehot, axis=1)

# map from labels to feature vectors
target_features = get_embeddings_from_labels(vision_labels, word_features)

inputs = keras.Input((vision_features.shape[1],))
associated_features = keras.layers.Dense(word_features.shape[1])(inputs)
associator = keras.Model(inputs, associated_features)
associator.compile('adam', 'mean_squared_error')

associator.fit(vision_features, target_features, batch_size=32, epochs=2)

associated_features.trainable = False
classifier_output = keras.layers.Dense(len(word_labels), activation='softmax')(
    associated_features)
combined_model = keras.Model(inputs, classifier_output)
combined_model.compile('adam', 'categorical_crossentropy', ['accuracy'])
combined_model.fit(vision_features, vision_labels_onehot, batch_size=32,
                   epochs=2)

predicted_word_features = associator.predict(vision_features)
predicted_labels = get_labels_from_embeddings(predicted_word_features,
                                              word_features)
reconstruction_accuracy = np.mean(predicted_labels == vision_labels)

print(reconstruction_accuracy)

np.save('./data/weights', combined_model.get_weights())
