# Vers une version inductive de t-SNE

Dans l'ère du big data et de l'apprentissage automatique, la visualisation et l'interprétation de données à haute dimensionnalité demeurent des défis majeurs. L'algorithme t-SNE (t-distributed Stochastic Neighbor Embedding), introduit par van der Maaten et Hinton en 2008, s'est imposé comme une technique de réduction de dimensionnalité et de visualisation de données particulièrement puissante. Cependant, malgré son efficacité, t-SNE présente certaines limitations, notamment son incapacité à généraliser à de nouvelles données sans recalculer l'ensemble de la projection.

Mon travail vise à implémenter en Python une approche prometteuse pour surmonter cette limitation : un t-SNE inductif basé sur l'utilisation d'un réseau de neurones profond, tel que décrit dans l'article de recherche de Roman-Rangel et Marchand-Maillet (2019) intitulé ["Inductive t-SNE via deep learning to visualize multi-label images"](https://www.sciencedirect.com/science/article/abs/pii/S0952197619300156). Cette méthode permet non seulement de conserver les avantages de t-SNE en termes de préservation de la structure locale des données, mais aussi d'offrir la capacité de projeter de nouveaux points dans l'espace réduit sans nécessiter un réapprentissage complet. Cette approche sera comparée à une méthode originale qui consiste à considérer le problème comme une tâche de régression qui sera réalisée avec XGBoost.

Le code n'est pas encore disponible, mais vous trouverez ci-dessous les premiers résultats de mon travail encore en cours.

## Implémentation en Python

```python
import numpy as np
from sklearn.manifold import TSNE
from tensorflow import keras
from keras import layers, optimizers, metrics
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split


class Inductive_TSNE_DNN(TSNE):
  """
  Classe pour t-SNE inductif utilisant un réseau de neurones profond.

  Cette classe étend l'algorithme t-SNE standard en utilisant un réseau de neurones
  pour apprendre une fonction de projection, permettant ainsi l'application de
  t-SNE à de nouvelles données sans réentraînement complet.

  Attributs:
    model (keras.Sequential): Le modèle de réseau de neurones.
    rmse (float): L'erreur quadratique moyenne entre la projection t-SNE originale
                  et celle prédite par le réseau de neurones.
    history (keras.callbacks.History): L'historique d'entraînement du modèle.
  """
  def __init__(self, n_components=2, **kwargs):
    """
    Initialise l'objet Inductive_TSNE_DNN.

    Args:
      n_components (int): Nombre de composantes pour la projection de sortie.
      **kwargs: Arguments supplémentaires passés à la classe parente TSNE.
    """
    super().__init__(n_components=n_components, **kwargs)
    self.model = None
    self.rmse = None
    self.history = None

  def _build_model(self, input_dim):
    """
    Construit l'architecture du réseau de neurones.

    Args:
      input_dim (int): Dimension des données d'entrée.

    Returns:
      keras.Sequential: Le modèle de réseau de neurones construit.
    """
    return keras.Sequential([
        layers.InputLayer(shape=(input_dim,)),
        layers.Dense(128, activation='relu', name='layer1'),
        layers.Dense(32, activation='relu', name='layer2'),
        layers.Dense(8, activation='relu', name='layer3'),
        layers.Dense(self.n_components, activation='linear', name='output')
    ])

  def fit(self, X, y=None):
    """
    Ajuste le modèle aux données d'entrée.

    Args:
      X (array-like): Données d'entrée de forme (n_samples, n_features).
      y: Ignoré, présent pour la compatibilité avec l'API scikit-learn.

    Returns:
      self: Retourne l'instance de l'objet.
    """
    # Effectue d'abord la projection t-SNE standard
    y_tsne = super().fit_transform(X)

    # Construit et compile le modèle
    self.model = self._build_model(X.shape[1])
    self.model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )

    # Entraîne le modèle
    self.history = self.model.fit(
        X, y_tsne,
        epochs=300,
        batch_size=min(32, len(X)),
        verbose=0,
        validation_split=0.1
      )

    # Calcule l'erreur RMSE
    y_pred = self.model.predict(X)
    self.rmse = np.sqrt(np.mean((y_tsne - y_pred) ** 2))

    return self

  def transform(self, X):
    """
    Applique la transformation aux nouvelles données.

    Args:
      X (array-like): Nouvelles données à transformer.

    Returns:
      array: Données transformées dans l'espace de faible dimension.
    """
    if self.model is None:
      raise ValueError("Le modèle n'a pas été entraîné. Appelez fit() d'abord.")
    return self.model.predict(X)

  def fit_transform(self, X, y=None):
    """
    Ajuste le modèle aux données et renvoie les données transformées.

    Args:
      X (array-like): Données d'entrée.
      y: Ignoré, présent pour la compatibilité avec l'API scikit-learn.

    Returns:
      array: Données transformées dans l'espace de faible dimension.
    """
    return self.fit(X).transform(X)

  def score(self, X, y=None):
    """
    Calcule le score du modèle (négatif de RMSE).

    Args:
      X (array-like): Données d'entrée.
      y: Ignoré, présent pour la compatibilité avec l'API scikit-learn.

    Returns:
      float: Score du modèle (plus élevé est meilleur).
    """
    if self.rmse is None:
      raise ValueError("Le modèle n'a pas été entraîné. Appelez fit() d'abord.")
    return -self.rmse

  def plot_training_history(self):
    if self.history is None:
      raise ValueError("Le modèle n'a pas été entraîné. Appelez fit() d'abord.")
    train_loss = self.history.history['loss']
    val_loss = self.history.history['val_loss']
    train_mae = self.history.history['mae']
    val_mae = self.history.history['val_mae']
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b', label='Perte d\'entraînement')
    plt.plot(epochs, val_loss, 'r', label='Perte de validation')
    plt.title('Perte d\'entraînement et de validation')
    plt.xlabel('Époques')
    plt.ylabel('Perte')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_mae, 'b', label='MAE d\'entraînement')
    plt.plot(epochs, val_mae, 'r', label='MAE de validation')
    plt.title('MAE d\'entraînement et de validation')
    plt.xlabel('Époques')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()
    plt.show()

class Inductive_TSNE_XGB(TSNE):
  """
  Classe pour t-SNE inductif utilisant XGBoost.

  Cette classe étend l'algorithme t-SNE standard en utilisant XGBoost
  pour apprendre une fonction de projection, permettant ainsi l'application de
  t-SNE à de nouvelles données sans réentraînement complet.

  Attributs:
    model (XGBRegressor): Le modèle XGBoost pour la régression multi-output.
    rmse (float): L'erreur quadratique moyenne entre la projection t-SNE originale
                  et celle prédite par XGBoost.
  """
  def __init__(self, n_components=2, **kwargs):
    """
    Initialise l'objet Inductive_TSNE_XGB.

    Args:
      n_components (int): Nombre de composantes pour la projection de sortie.
      **kwargs: Arguments supplémentaires passés à la classe parente TSNE.
    """
    super().__init__(n_components=n_components, **kwargs)
    self.model = None
    self.rmse = None

  def fit(self, X, y=None):
    """
    Ajuste le modèle aux données d'entrée.

    Cette méthode effectue d'abord une projection t-SNE standard,
    puis entraîne un modèle XGBoost pour apprendre cette projection.

    Args:
      X (array-like): Données d'entrée de forme (n_samples, n_features).
      y: Ignoré, présent pour la compatibilité avec l'API scikit-learn.

    Returns:
      self: Retourne l'instance de l'objet.
    """
    # Effectue d'abord la projection t-SNE standard
    y_tsne = super().fit_transform(X)

    # Crée et entraîne le modèle XGBoost
    self.model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.5,
        eval_metric='rmse',
        random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_tsne,
        test_size=0.1,
        random_state=42
    )
    self.model.fit(
        X, y_tsne,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )

    # Calcule l'erreur RMSE
    y_pred = self.model.predict(X)
    self.rmse = np.sqrt(np.mean((y_tsne - y_pred) ** 2))

    return self

  def transform(self, X):
    """
    Applique la transformation aux nouvelles données.

    Args:
      X (array-like): Nouvelles données à transformer.

    Returns:
      array: Données transformées dans l'espace de faible dimension.

    Raises:
      ValueError: Si le modèle n'a pas été entraîné.
    """
    if self.model is None:
      raise ValueError("Le modèle n'a pas été entraîné. Appelez fit() d'abord.")
    return self.model.predict(X)

  def fit_transform(self, X, y=None):
    """
    Ajuste le modèle aux données et renvoie les données transformées.

    Args:
      X (array-like): Données d'entrée.
      y: Ignoré, présent pour la compatibilité avec l'API scikit-learn.

    Returns:
      array: Données transformées dans l'espace de faible dimension.
    """
    return self.fit(X).transform(X)

  def score(self, X, y=None):
    """
    Calcule le score du modèle (négatif de RMSE).

    Args:
      X (array-like): Données d'entrée.
      y: Ignoré, présent pour la compatibilité avec l'API scikit-learn.

    Returns:
      float: Score du modèle (plus élevé est meilleur).

    Raises:
      ValueError: Si le modèle n'a pas été entraîné.
    """
    if self.rmse is None:
      raise ValueError("Le modèle n'a pas été entraîné. Appelez fit() d'abord.")
    return -self.rmse
```

## Résultats sur le jeu de données Swiss Roll.

Le Swiss Roll est un jeu de données synthétique tridimensionnel qui ressemble, comme son nom l'indique, à un rouleau suisse ou à une spirale 3D. Il est créé mathématiquement pour simuler une structure de données non linéaire dans un espace à haute dimension. Le Swiss Roll est localement plat mais non linéaire ce qui en fait un très bon test pour les algorithmes de réductions de dimensionalité.

Paramètres du Swiss Roll:
  * Bruit: 0.1
  * Nombre de points: 4000

![image](https://github.com/user-attachments/assets/5dddf43a-babd-419e-947d-88f328b6a74b)
![image](https://github.com/user-attachments/assets/47fa505f-5784-4083-968b-0f957f280fb2)
![image](https://github.com/user-attachments/assets/38ac3e23-1ece-4a0a-b663-ec722799cc57)
![image](https://github.com/user-attachments/assets/3b15f5a5-648a-4538-8dc1-045e6d247945)
![image](https://github.com/user-attachments/assets/f8e81bfa-a85f-4eb6-b8eb-1dcfa2c98385)

Les premiers résultats sont très prometteurs pour les deux méthodes. La capacité de XGBoost à reproduire la structure des données en 2D produite par t-SNE est cependant nettement supérieure à celle du réseau de neurones profond, mais probablement au détriment d'une capacité de généralisation moindre. L'écart des durées d'apprentissage est significatif sur des jeux de données de petite taille mais semble se réduire significativement à mesure que le nombre de points augmente dans le jeu d'entraînement.

## A venir

Afin d'étudier les capacités de généralisation des deux algorithmes, on mesurera leurs performances respectives sur des jeux de données plus complexes comme MNIST et Digits.
