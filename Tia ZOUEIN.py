#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install xgboost


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras.models import Model
from keras.layers import Input, Dense, Masking, SimpleRNN, GlobalAveragePooling1D, Dropout
from keras.layers import concatenate
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Input, Reshape, LSTM, Flatten, concatenate
from keras.models import Model
from keras.optimizers import Adam
import numpy as np

import xgboost as xgb
from xgboost import XGBClassifier


# J'ai décidé de fusionner les deux bases de données du match_1 et du match_2 pour former une seule base de données, vue qu'elles contiennent les mêmes données.

# In[2]:


df_1 = pd.read_json("match_1.json")
df_2 = pd.read_json("match_2.json")
df = pd.concat([df_1, df_2], ignore_index=True)
df


# **valeurs manquantes:**

# In[3]:


df.isnull().sum()


# Comme indiqué dans l'annonce du test, la variable "norm" contient des séquences de longueurs variables :

# In[4]:


df['norm'].apply(lambda x: len(x)).unique()


# In[5]:


df["label"].value_counts()


# On remarque que les actions les plus fréquentes dans la base de données sont "run" et "walk". D'autre part, les actions "cross" et "no actions" sont rares. On peut visualiser ces résultats:

# In[6]:


plt.figure(figsize=(10, 6))
sns.countplot(x='label', data=df)
plt.title('Distribution des actions')
plt.show()


# Dans ce qui suit, je calcule la norme moyenne dans chaque séquence, puis la norme moyenne par action. On remarque que la vitesse moyenne de l'action "cross" est presque la même que celle de l'action "dribble". L'action "shot" a la vitesse moyenne la plus élevée, et l'action "rest" la plus faible.

# In[7]:


df['norm_mean'] = df['norm'].apply(lambda x: sum(x) / len(x))
result = df.groupby("label")['norm_mean'].mean().reset_index()
result


# In[8]:


sns.lineplot(x='label', y='norm_mean', data=result)
plt.tight_layout()
plt.show()


# Dans la suite, je trace les valeurs de la norme moyenne pour chaque action. Cela nous permet de visualiser la variabilité de la vitesse moyenne d'une observation à l'autre pour une même action.

# In[9]:


for variable in df["label"].unique():
    subset = df[df['label'] == variable]
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=subset.index, y='norm_mean', data=subset)
    plt.title(f'Norm moyenne pour l\'action "{variable}"')
    plt.xlabel(f'Index des observations correspondantes à l\'action "{variable}"')
    plt.ylabel('Norme')
    plt.show()


# On peut aussi calculer la variance de la norme pour chaque action.

# In[10]:


variance_by_action = df.groupby('label')['norm_mean'].var()
print(variance_by_action)


# Dans la suite, je calcule la longueur de chaque séquence dans la variable "norm", puis je calcule la longueur moyenne par action. On remarque que les actions 'no action' et 'rest' ont les séquences les plus longues, alors que les autres actions impliquant un mouvement ont des longueurs similaires.

# In[11]:


df['longueur_norm'] = df['norm'].apply(len)
result = df.groupby("label")['longueur_norm'].mean().reset_index()
result


# In[12]:


result["longueur_norm"].mean()


# **Recréation du jeu**

# **Generative Adversial Network**

# Le GAN (Generative Adversial Network) est un modèle de machine learning qui vise à générer de nouvelles données qui ressemblent aux données initiales. Un GAN est composé de deux réseaux principaux :
# 
# -Le Générateur : Il prend en entrée un vecteur aléatoire et génère des données qui ressemblent à celles de l'ensemble d'entraînement.
# 
# -Le Discriminateur : Il prend en entrée des données réelles et générées par le générateur, puis tente de les distinguer l'une de l'autre.
# 
# Le générateur et le discriminateur sont entraînés simultanément de manière itérative.
# 
# La fonction coût du générateur mesure à quel point il est capable de tromper le discriminateur en générant des données qui ressemblent à des données réelles. Elle est donnée par la formule suivante: 
# $$
# J(G) = -\frac{1}{2} \mathbb{E}_{x \sim p_{\text{data}}(x)} \left[ \log D(G(x)) \right]
# $$
# avec : D(G(x)) est la sortie du discriminateur pour une donnée générée G(x) et p(x) est la distribution du bruit d'entrée.
# 
# La fonction coût du discriminateur mesure à quel point il est capable de distinguer entre les données réelles et générées. Elle est donnée par:
# $$
# J(D) = -\frac{1}{2} \mathbb{E}_{x \sim p_{\text{data}}(x)} \left[ \log D(x) \right] - \frac{1}{2} \mathbb{E}_{z \sim p_z(z)} \left[ \log(1 - D(G(z))) \right]
# $$
# - \(D(x)\) est la sortie du discriminateur pour une donnée réelle \(x\).
# - \(G(z)\) est la sortie du générateur pour un bruit d'entrée \(z\).
# - \(p_{data}(x)\) est la distribution des données réelles.
# - \(p_z(z)\) est la distribution du bruit d'entrée.
# 
# La fonction de coût totale est définie comme la somme de ces deux termes. Les signes négatifs dans les fonctions de coût sont utilisés car il s'agit d'une maximisation du log-likelihood pour le discriminateur et d'une minimisation pour le générateur.
# 

# J'ai donc choisis d'utiliser les GAN pour reproduire le mouvement d'un joueur pendant un match de football. J'ai trouvé que les GAN particulièrement adaptés à cette tâche car ils offrent la possibilité d'apprendre la distribution des données présentes dans mon ensemble de données actuel, tout en étant capables de générer de nouvelles données qui conservent la cohérence des mouvements observés. De plus, il permet de répondre à l'exigence demandée que l'output du modèle doit etre sous le meme format de l'input, sous forme d'un dictionnaire.

# **Construction du générateur:**

# In[13]:


def build_generator(latent_dim, num_classes, max_norm_length):
    action_input = Input(shape=(latent_dim,))
    norm_input = Input(shape=(max_norm_length,))
    
    # obtention des actions en sortie du générateur
    action_output = Dense(num_classes, activation='softmax')(action_input)
    # obtention des normes en sortie du générateur
    norm_reshaped = Reshape((1, max_norm_length))(norm_input)
    norm_lstm = LSTM(128, return_sequences=True, activation='linear')(norm_reshaped)
    norm_flatten = Flatten()(norm_lstm)

    # Concaténation des actions et des normes
    concatenated = concatenate([action_output, norm_flatten])

    generator_output = Dense(max_norm_length + num_classes, activation='linear')(concatenated)
    generator = Model(inputs=[action_input, norm_input], outputs=generator_output)
    return generator


# Les données générées sont constituées d'actions et de normes d'accélération. Ces données sont ensuite utilisées pour entraîner le discriminateur à distinguer les données réelles des données générées.

# Vue qu'on peut traiter les données comme une série temporelle, j'ai utilisé la couche LSTM (Long Short-Term Memory). Elle permet d'apprendre les motifs séquentiels dans les données, aidant ainsi à générer des séquences temporelles réalistes.

# **Construction du discriminateur**

# In[14]:


def build_discriminator(input_dim, max_norm_length):
    model = Sequential()
    model.add(LSTM(128, input_shape=(max_norm_length, 1), return_sequences=True))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    discriminator_input = Input(shape=(input_dim, 1))
    validity = model(discriminator_input)
    discriminator = Model(inputs=discriminator_input, outputs=validity)
    return discriminator


# **Construction du GAN**:

# In[15]:


def build_gan(generator, discriminator, latent_dim, max_norm_length):
    discriminator.trainable = False

    gan_action_input = Input(shape=(latent_dim,))
    gan_norm_input = Input(shape=(max_norm_length,))
    generated_data = generator([gan_action_input, gan_norm_input])

    # Ajustez la couche de remodelage pour s'adapter à la longueur maximale attendue par le discriminateur
    reshaped_data = Reshape((max_norm_length, 1))(Dense(max_norm_length)(generated_data))
    validity = discriminator(reshaped_data)

    gan = Model(inputs=[gan_action_input, gan_norm_input], outputs=validity)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

    return gan

latent_dim = 100
num_classes = df["label"].nunique()
max_norm_length = max(df['norm'].apply(len))

generator = build_generator(latent_dim, num_classes, max_norm_length)
generator.compile(loss='categorical_crossentropy', optimizer='adam')


discriminator = build_discriminator(max_norm_length, max_norm_length)
discriminator.compile(loss='binary_crossentropy', optimizer='adam')

gan = build_gan(generator, discriminator, latent_dim, max_norm_length)


generator.summary()
discriminator.summary()
gan.summary()


# **Entraînement du modèle**:

# La fonction "generate_fake_data" génère des données non réelles en créant des actions et des normes d'accélération aléatoires. Ces données sont utilisées pour entraîner le générateur.
# D'autre part, la fonction "generate_real_data" extrait des données réelles du jeu d'entraînement. Elle choisit aléatoirement des séquences d'accélération existantes dans l'ensemble de données, remplit les séquences pour avoir la même longueur, et utilise les labels correspondants. Ces données sont utilisées pour entraîner le discriminateur à reconnaître les données réelles.

# In[16]:


# Fonction pour générer des données d'entraînement factices

def generate_fake_data(batch_size, latent_dim, num_classes, max_norm_length):
    fake_actions = np.random.rand(batch_size, latent_dim)
    fake_norms = np.random.rand(batch_size, max_norm_length)
    fake_labels = np.random.randint(num_classes, size=batch_size)
    fake_data = [fake_actions, fake_norms]
    fake_data_labels = np.zeros((batch_size, 1))

    return fake_data, fake_data_labels

def generate_real_data(batch_size, training_data, max_norm_length):
    indices = np.random.randint(0, len(training_data), size=batch_size) % len(training_data)
    real_data = [np.array(df.iloc[i]['norm']) for i in indices]  # Convertir la liste en tableau numpy
    real_labels = [df.iloc[i]['label'] for i in indices]

    real_data = [np.concatenate([arr, np.zeros(max_norm_length - len(arr))]) for arr in real_data]

    real_data = np.array(real_data)
    real_labels = np.array(real_labels)

    real_data_labels = np.ones((batch_size, 1))

    return real_data, real_data_labels

# Pour s'assurer que toutes les séquences ont la même longueur, j'ai utilisé une technique de padding. 
#Cela implique d'ajouter des valeurs (j'ai choisis 0) à la fin de chaque séquence pour qu'elles atteignent une longueur commune. 
def train_gan(generator, discriminator, gan, training_data, epochs, batch_size, latent_dim, num_classes, max_norm_length):
    for epoch in range(epochs):
        for _ in range(len(training_data) // batch_size):
            real_data, real_data_labels = generate_real_data(batch_size, training_data, max_norm_length)

            real_data = pad_sequences(real_data, maxlen=max_norm_length, padding='post', dtype='float32')
            discriminator_loss_real = discriminator.train_on_batch(real_data[:, :, np.newaxis], real_data_labels)

            fake_data, fake_data_labels = generate_fake_data(batch_size, latent_dim, num_classes, max_norm_length)
            generated_data = generator.predict(fake_data)

            generated_data = pad_sequences(generated_data, maxlen=max_norm_length, padding='post', dtype='float32')
            discriminator_loss_fake = discriminator.train_on_batch(generated_data[:, :, np.newaxis], fake_data_labels)

            discriminator_loss = 0.5 * (discriminator_loss_real + discriminator_loss_fake)

            gan_labels = np.ones((batch_size, 1))
            generator_loss = gan.train_on_batch(fake_data, gan_labels)

        print(f"Epoch {epoch + 1}, Discriminator Loss: {discriminator_loss}, Generator Loss: {generator_loss}")

    return gan

epochs = 30
batch_size = 16

gan = train_gan(generator, discriminator, gan, df, epochs, batch_size, latent_dim, num_classes, max_norm_length)


# In[49]:


def generate_samples(generator, latent_dim, num_samples, max_norm_length):
    fake_actions = np.random.rand(num_samples, latent_dim)
    fake_norms = np.random.rand(num_samples, max_norm_length)

    fake_labels = np.argmax(generator.predict([fake_actions, fake_norms]), axis=-1) 

    return {'norm': fake_norms, 'label': fake_labels}

max_norm_length = max(df['norm'].apply(len))

num_generated_samples = 25
generated_samples = generate_samples(generator, latent_dim, num_generated_samples, max_norm_length)
label_mapping = {i: label for i, label in enumerate(df['label'].unique())}

for i in range(num_generated_samples):
    print("Generated Sample", i+1)
    print("Norm:", generated_samples['norm'][i])
    generated_label_index = np.argmax(generated_samples['label'][i], axis=-1)
    generated_label = label_mapping[generated_label_index]
    print("Label:", generated_label)
    print("\n")


# ## **La deuxième approche:**

# Ma deuxième approche pour recrééer le jeu est basée sur une architecture de réseau de neurones récurrents (RNN) spécifiquement, une couche LSTM (Long Short-Term Memory). Mon choix se justifie par la nature séquentielle des données d'accélération que je traite.
# 
# Le modèle que j'ai construit prend en entrée les normes, et a pour objectif de prédire l'action associée à chaque séquence. Pour préparer les données, j'ai calculé la vitesse moyenne de toutes les séquences d'accélération pour être utilisée comme valeur au lieu de 0 pour remplir les séquences, et j'ai utilisé le LabelEncoder pour convertir les labels d'actions en valeurs numériques.
# 
# Il commence par une couche de masquage (Masking) qui permettra de masquer les valeurs 0 qu'on a ajouté avec pad_sequences pour gérer les séquences de longueur variable, suivie d'une couche LSTM pour capturer les dépendances temporelles dans les séquences. J'ai ajouté des couches de dropout pour régulariser le modèle et éviter le surajustement. 
# 
# Enfin, le modèle est entraîné sur les données d'entraînement avec une validation sur un sous-ensemble de données. Les prédictions des actions sont générées sur l'ensemble de test.

# In[23]:


mean_speed = np.mean([np.mean(seq) for seq in df['norm']])
sequences = df['norm'].tolist()

padded_sequences = pad_sequences(sequences, padding='post', dtype='float32', value=mean_speed)

labels = df['label'].tolist()

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

y_train_encoded = to_categorical(encoded_labels)

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, y_train_encoded, test_size=0.2, random_state=42)

model = Sequential()
model.add(Masking(mask_value=mean_speed, input_shape=(None, 1)))
model.add(LSTM(128))

model.add(Dropout(0.05))
model.add(Dense(64, activation='relu'))

model.add(Dropout(0.05))
model.add(Dense(32, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=20, validation_split = 0.1)

predictions = model.predict(X_test)

recreated_game = [{'norm': sequence.tolist(), 'label': label_encoder.classes_[np.argmax(prediction)]} for sequence, prediction in zip(padded_sequences, predictions)]
recreated_game


# In[29]:


for prediction in predictions:
    print(label_encoder.classes_[np.argmax(prediction)] )


# In[30]:


plt.plot(history.history['loss'], label='training set',marker='o', linestyle='solid',linewidth=1, markersize=6)
plt.plot(history.history['val_loss'], label='validation set',marker='o', linestyle='solid',linewidth=1, markersize=6)
plt.title("model loss")
plt.xlabel('#Epochs')
plt.ylabel('Total Loss')
plt.legend(bbox_to_anchor=( 1.35, 1.))


# J'ai créée cette fonction generate_games pour synthétiser des jeux de données en utilisant mon modèle de prédiction d'actions ci-dessus. Elle permet de choisir aléatoirement des séquences de la base de données, de les ajuster à la longueur souhaitée pour un jeu, de prédire les actions correspondantes à l'aide du modèle, et de créer ainsi des jeux générés avec des séquences et leurs labels d'action associés.

# In[38]:


import numpy as np

def generate_games(model, label_encoder, num_generated_games=15, game_length=10, game_type='normal'):
    generated_games = []

    for _ in range(num_generated_games):
        random_index = np.random.randint(0, len(df))
        generated_sequence = df.iloc[random_index]['norm']
        
        while len(generated_sequence) < game_length:
            generated_sequence += generated_sequence

        start_index = np.random.randint(0, len(generated_sequence) - game_length + 1)
        selected_sequence = generated_sequence[start_index:start_index + game_length]
        
        generated_sequence = np.array(selected_sequence).reshape(1, -1, 1)

        generated_prediction = model.predict(generated_sequence)

        generated_game = {'norm': selected_sequence,
                          'label': label_encoder.classes_[np.argmax(generated_prediction)]}

        generated_games.append(generated_game)

    return generated_games


num_generated_games = 10
game_length = 61
game_type = 'normal'

generated_games = generate_games(model, label_encoder, num_generated_games, game_length, game_type)

for i, game in enumerate(generated_games):
    print(f"Jeu généré {i + 1}: {game}")


# La deuxième approche se base sur les normes pour prédire l'action. Donc la norme n'est pas générée par le modèle. Dans ce qui suit, j'ai tenté à essayé une approche qui consiste à construire un modèle hybride qui prend en compte à la fois les données d'accélération (normes) et les actions associées. L'objectif est de prédire à la fois l'action du joueur et les normes d'accélération correspondantes.
# 
# Les séquences d'accélération sont traitées en utilisant la fonction pad_sequences pour les remplir avec des zéros et les ajuster à la même longueur. Cela garantit que toutes les séquences ont la même longueur, ce qui est essentiel pour l'entrée du modèle. D'autre part, les étiquettes d'actions encodées sont converties en représentation one-hot à l'aide de to_categorical. Cette représentation permet au modèle de comprendre les relations entre les différentes actions.
# 
# J'ai choisi une architecture de modèle avec deux couches distinctes pour traiter les données d'accélération et d'actions: une couche de masquage (Masking) comme avant, suivie d'une couche RNN (SimpleRNN) pour capturer les motifs séquentiels. La couche GlobalAveragePooling1D pour traiter les séquences de longueur variable en calculant la moyenne sur toute la séquence. Cela permet de générer une représentation fixe pour chaque séquence pour éviter tout erreur de code.

# In[16]:


label_encoder = LabelEncoder()

df['encoded_label'] = label_encoder.fit_transform(df['label'])
num_classes = len(label_encoder.classes_)


X_norms = pad_sequences(df['norm'], padding='post', dtype='float32', truncating='post')
y_norms = pad_sequences(df['norm'], padding='post', dtype='float32', truncating='post')


X_actions = to_categorical(label_encoder.transform(df['label']), num_classes=num_classes)
y_actions = to_categorical(label_encoder.transform(df['label']), num_classes=num_classes)


X_norm_train, X_norm_test, X_actions_train, X_actions_test, y_actions_train, y_actions_test, y_norms_train, y_norms_test = train_test_split(
    X_norms, X_actions, y_actions, y_norms, test_size=0.2, random_state=42
)

input_layer_norm = Input(name='norm_input', shape=(X_norms.shape[1], 1))
input_layer_actions = Input(name='action_input', shape=(X_actions.shape[1], 1))

masking_layer_norm = Masking(mask_value=0.0)(input_layer_norm)
masking_layer_actions = Masking(mask_value=0.0)(input_layer_actions)

rnn_layer_norm = SimpleRNN(128, activation='relu', return_sequences=True)(masking_layer_norm)
rnn_layer_actions = SimpleRNN(128, activation='relu', return_sequences=True)(masking_layer_actions)

pooled_layer_norm = GlobalAveragePooling1D()(rnn_layer_norm)
pooled_layer_actions = GlobalAveragePooling1D()(rnn_layer_actions)

dense_layer_1 = Dense(64, activation='relu')(pooled_layer_actions)
dropout_layer_1 = Dropout(0.3)(dense_layer_1)
dense_layer_2 = Dense(32, activation='relu')(dropout_layer_1)
dropout_layer_2 = Dropout(0.2)(dense_layer_2)
dense_layer_3 = Dense(16, activation='relu')(dropout_layer_2)
dropout_layer_3 = Dropout(0.1)(dense_layer_3)

dense_norm_layer_1 = Dense(64, activation='relu')(pooled_layer_norm)
dropout_norm_layer_1 = Dropout(0.3)(dense_norm_layer_1)
dense_norm_layer_2 = Dense(32, activation='relu')(dropout_norm_layer_1)
dropout_norm_layer_2 = Dropout(0.2)(dense_norm_layer_2)
dense_norm_layer_3 = Dense(16, activation='relu')(dropout_norm_layer_2)
dropout_norm_layer_3 = Dropout(0.1)(dense_norm_layer_3)


action_output = Dense(num_classes, activation='softmax', name='action_output')(dropout_layer_3)

norm_output = Dense(X_norms.shape[1], activation='linear', name='norm_output')(dropout_norm_layer_3)

model = Model(inputs={'norm_input': input_layer_norm, 'action_input': input_layer_actions},
              outputs=[action_output, norm_output])

model.compile(loss={'action_output': 'categorical_crossentropy', 'norm_output': 'mean_squared_error'},
              optimizer='adam',
              metrics={'action_output': 'accuracy', 'norm_output': 'mae'})

history = model.fit({'norm_input': X_norm_train, 'action_input': X_actions_train},
                    {'action_output': y_actions_train, 'norm_output': y_norms_train},
                    epochs=60, batch_size=50, validation_split=0.1)


# ## **Dernière approche**

# La dernière approche consiste à extraire toutes les caractéristiques statistiques possibles de la variable norme, et la remplace par ces informations. Ensuite, on peut prédire l'action et le mouvement du joueur, et on peut recrééer le jeu en utilisant la fonction generate_games définie ci-dessus. Cependant, cette approche ne permet pas de prédir l'action et la norme en meme temps, pour cela j'ai juste développée l'idée de cette approche, vue que le but est de recrééer le jeu et de générer à la fois l'action avec sa norme correspondante. 

# In[50]:


df['norm_mean'] = df['norm'].apply(lambda x: pd.Series(x).mean())

df['norm_std'] = df['norm'].apply(lambda x: pd.Series(x).std())

df['norm_median'] = df['norm'].apply(lambda x: pd.Series(x).median())

df['norm_min'] = df['norm'].apply(lambda x: pd.Series(x).min())

df['norm_max'] = df['norm'].apply(lambda x: pd.Series(x).max())

df['norm_var'] = df['norm'].apply(lambda x: pd.Series(x).var())

df['norm_q1'] = df['norm'].apply(lambda x: pd.Series(x).quantile(0.25))

df['norm_q2'] = df['norm'].apply(lambda x: pd.Series(x).quantile(0.5))

df['norm_q3'] = df['norm'].apply(lambda x: pd.Series(x).quantile(0.75))

df['norm_iqr'] = df['norm'].apply(lambda x: pd.Series(x).quantile(0.75) - pd.Series(x).quantile(0.25))

df['norm_range'] = df['norm'].apply(lambda x: pd.Series(x).max() - pd.Series(x).min())

df


# In[51]:


X = df.drop(["norm", "label"], axis=1)
Y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[52]:


xgboost_model = xgb.XGBClassifier(learning_rate=0.1,
                                  gamma=0.1,
                                  max_depth=5,
                                  min_child_weight=2,
                                  subsample=0.7,
                                  reg_lambda=1,
                                  alpha=0.1,
                                  n_estimators=50)


# **Optimisation des hyperparamètres**:

# In[51]:


import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score

param_grid = {
    'eta': [0.001, 0.01, 0.1],
    'gamma': [0, 1, 2, 5, 10, 20],
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 2, 3],
    'subsample': [0.8, 0.9, 1.0],
    'reg_lambda': [0, 0.1, 0.2],
    'alpha': [0, 0.1, 0.2],
    'n_estimators': [30, 50, 80]
}

random_search = RandomizedSearchCV(estimator=xgboost_model, param_distributions=param_grid, scoring='accuracy', cv=10)

# Exécuter la recherche par grille sur les données
random_search.fit(X_train, y_train)

# Afficher les meilleurs hyperparamètres
print("Meilleurs hyperparamètres:", random_search.best_params_)


# In[52]:


# Utiliser le modèle avec les meilleurs hyperparamètres pour faire des prédictions
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)


# In[53]:


# Évaluer la performance du modèle
accuracy = accuracy_score(y_test, y_pred)
print("Précision du modèle optimisé:", accuracy)


# In[54]:


conf_matrix = confusion_matrix(y_test , y_pred)
conf_matrix


# In[ ]:


recreated_game_xgboost = [{'norm': X_test.iloc[i, :].tolist(), 'label': label} for i, label in enumerate(y_pred)]


# In[ ]:




