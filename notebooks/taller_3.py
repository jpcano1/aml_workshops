# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="wy53BRdZ0duK"
# ## **Taller 3 - Clasificación de Géneros**
# - Juan Pablo Cano - 201712395

# %% colab={"base_uri": "https://localhost:8080/", "height": 84, "referenced_widgets": ["ad45766a7d2842af82555e34fdab4fcf", "e2650135ecd643888d99338b91bb5e7e", "cb2c5143856644288e897c17f276ede1", "6808ddca96b44f7aa318e568ba21da3b", "bfc1a65c84434f5fb4704580678c8acc", "4f2421d4d5b745e89c7ca1fe0ffb2dc7", "0e59efe5427847358ba9633846901143", "f341beaf1c084ce9adaeb4dac3ab2baf", "b4ea8609b5374f38a747e9ef2e8c36d6", "e4c76ca79ab54fc2bf57f35d1031b7f3", "8563f6c8fd944f4f806fb6f6f80a678a"]} id="NK4c5k8RIAJf" outputId="26d51245-f4ad-40af-959a-fbf0937223c6"
# !shred -u setup_colab_general.py
# !wget -q "https://github.com/jpcano1/python_utils/raw/main/setup_colab_general.py" -O setup_colab_general.py
import setup_colab_general as setup_general

setup_general.setup_general()

# %% colab={"base_uri": "https://localhost:8080/"} id="d6IOX7s8CvMt" outputId="8e746392-9d15-463c-a777-b303230c0c74"
# !pip install -q loguru
# !pip install -q -U gdown
# !pip install -q tensorflow-addons
# !pip install -q -U keras-tuner

# %% [markdown] id="7ryBvc710gHS"
# ## **Importando las librerías necesarias para el taller**

import matplotlib.pyplot as plt

plt.style.use("seaborn-deep")

import functools
import os
from typing import ByteString, List, Optional, Tuple, Union

import keras_tuner as kt
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import tensorflow_addons as tfa
from loguru import logger
from skimage import io
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow import keras
from tqdm.auto import tqdm

from .utils import general as gen

# %% id="9ykCXVE19Mvc"
read_image = lambda x: io.imread(x)
extract_label = lambda x: "|".join(x)

# %% id="J5pAliZME3-i"
BASE_PATH = "data/images"

# %% colab={"base_uri": "https://localhost:8080/"} id="U8gH4O8OGuU8" outputId="594924e6-a64c-4fde-f14e-271f2a3c8d63"
# %%shell
# mkdir data/
gdown -q 1anzSoHA4J0QjfrH8oeCVjSqtTjrSmR7H -O data/movie_genre.csv

# %% [markdown] id="8p-YW0ej0kqU"
# ## **Lectura de Datos y Visualización**
# - Tenemos un dataset en CSV que contiene los géneros y las URLS de cada imagen.
# - La primera labor es analizar en busca de datos nulos en el dataset.

# %% id="wSk_UV0sIqRW"
data = pd.read_csv("data/movie_genre.csv", encoding="ISO-8859-1")

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="wwxj3D1rrKL3" outputId="42163cb2-d36c-4d94-c706-4884d8f210c7"
data.head()

# %% colab={"base_uri": "https://localhost:8080/"} id="qF4YkOQIrR99" outputId="464dfc9a-b983-4bd5-e2fa-02c87df2f9af"
data.shape

# %% colab={"base_uri": "https://localhost:8080/"} id="qjIx8CH8Lfgt" outputId="98765b62-bd3c-4ab1-c3ce-25fa072006bd"
data.columns

# %% [markdown] id="7cU_4_Gk0m7q"
# ### **Limpieza y Análisis**

# %% id="-jtvKX8fMei-"
data.drop(columns=["Title", "IMDB Score", "Imdb Link"], inplace=True)
data.columns = ["id", "genre", "link"]

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="iFo7nbbSH510" outputId="ae7d6aef-f074-4890-94c6-5538cbc53085"
data.head()

# %% [markdown] id="DANiYWBt3op0"
# - Es necesario eliminar cualquier dato nulo asociado a la URL de la imagen o a su etiqueta. De lo contrario, el entrenamiento será imposible.

# %% id="ftf89WlbPVxQ"
data = data.dropna(subset=["link", "genre"]).reset_index(drop=True)

# %% colab={"base_uri": "https://localhost:8080/"} id="xGv8hjstrURn" outputId="5a4ae7b1-2c34-4556-8218-7986b8b61694"
data.shape

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="SoPETHpDPaO-" outputId="1db8a1fe-9d73-44d4-98f4-5a6b3b04aa2c"
data.head()

# %% id="VbNIFqLuPVPu"
counts = (
    data["genre"]
    .apply(lambda x: x.split("|"))
    .explode()
    .value_counts()
    .sort_values(ascending=False)
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 303} id="hIfUzh_hPpnV" outputId="10c4cfc6-fb6c-47be-ff2e-cbe4bee1422c"
plt.figure(figsize=(35, 5))
plt.bar(x=counts.index, height=counts.values)
plt.title("Label Distribution")
plt.show()

# %% [markdown] id="hChoqpMS3uAw"
# - Para quitarnos el ruido de los géneros con pocas imágenes, eliminamos aquellos con menos de 1100 incidencias. Al final quedamos con 18 etiquetas.

# %% id="cCd7AUm2P-CM"
labels_to_drop = counts[counts < 1100]

# %% id="wAMwbDcdQIw-"
data["genre"] = data["genre"].apply(
    lambda x: [part for part in x.split("|") if part not in labels_to_drop]
)

# %% colab={"base_uri": "https://localhost:8080/"} id="1MOBgsOYsQc0" outputId="6af3e1db-992e-4107-f8ee-29f853a752bd"
data.shape


# %% [markdown] id="ZLAD1SVp0rQV"
# ### **Descarga de los Datos**
# - Procedemos a descargar los datos.

# %% id="miMq-1VVRchR"
def get_image(link: str, file_path: str) -> bool:
    """
    Download a single image from URL.

    :param link: The link to download the image
    :param file_path: The file of the image to be stored
    :return: A flag to know if the file was, indeed, downloaded
    """
    try:
        r = requests.get(
            link,
            timeout=2,
            hooks={"response": check_response},
        )
        with open(file_path, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        logger.error(e)
        return False


def download_images(movies_df: pd.DataFrame, dst: str = "data") -> List[int]:
    """
    Download images from dataframe.

    :param movies_df: The dataframe with the links of the images
    :param dst: The destination directory for the images
    :return: Returns a list with the ids that were correctly downloaded
    """
    logger.info(f"Download started")

    if not os.path.exists(os.path.join(dst, "images")):
        os.makedirs(os.path.join(dst, "images"))

    correct_ids = []
    for id_, url in tqdm(movies_df[["id", "link"]].values):
        filename = os.path.join(dst, "images", f"{id_}.jpg")
        correct = get_image(url, filename)
        if correct:
            correct_ids.append(id_)

    logger.info("Download Finished")
    return correct_ids


def check_response(response: requests.Response, **kwargs) -> None:
    """
    Check response status code as hook.

    :param response: The response of the request performed
    """
    if not response.ok:
        raise RuntimeError(
            f"Error in request:\n"
            f"method: {response.request.method} - url: {response.url}\n"
            f"status code: {response.status_code}"
        )


# %% colab={"base_uri": "https://localhost:8080/", "height": 761, "referenced_widgets": ["257021a6a1014f57a45091a96ae71cf5", "2fbaffe92d324a5a8271abe189ee65a3", "e9cf62e6b4cc403abb81b4d2cc704c79", "72da5a351410482c91b8fb0c1db547c4", "126bdb24d5904a62a3f56853806439b6", "277ce1a43ed84edb93cd48838a9f0d9c", "a133bc484765435791558139318bffed", "318af24418c44f69a7e21bd5fa5f5414", "25ca0da56b8449e9a0eb22a48a12426f", "4acbbec6a8c44a8889f994dff32ea9c2", "f1ec811ea1664d60bad2697422e5f6c5"]} id="CG8Xc5icULA5" outputId="adf5b0aa-1031-4f33-ab7d-d8f0410ab399"
correct_ids = download_images(data)

# %% id="F-Dl8Rnvg08T"
data = data[data["id"].isin(correct_ids)].reset_index(drop=True)

# %% colab={"base_uri": "https://localhost:8080/"} id="GhOnqTP3juPd" outputId="5c52c7ac-5290-402a-c104-ae57ee5b1586"
data.shape

# %% id="iu8ObsTtkuGK"
data["file_path"] = data["id"].apply(lambda x: os.path.join("data", "images", f"{x}.jpg"))

# %% colab={"base_uri": "https://localhost:8080/", "height": 536} id="MyLxdYA6-Kty" outputId="780561fc-1224-4f26-b404-0cbe93764a62"
image_index = np.random.choice(data.index, 9)

images = list(map(read_image, data.iloc[image_index]["file_path"]))
labels = list(map(extract_label, data.iloc[image_index]["genre"]))

gen.visualize_subplot(
    imgs=images,
    titles=labels,
    division=(3, 3),
    figsize=(9, 9),
)

# %% [markdown] id="BqKXr73B3706"
# - Particionamos los datasets en entrenamiento, validación y testing.

# %% id="6B-RdLffjzGD"
full_X_train, X_test, full_y_train, y_test = train_test_split(
    data["file_path"].values,
    data["genre"].values,
    random_state=1234,
    test_size=0.2,
)
X_train, X_val, y_train, y_val = train_test_split(
    full_X_train, full_y_train, random_state=1234, test_size=0.2
)

# %% colab={"base_uri": "https://localhost:8080/"} id="BYF7Wj6aqZvb" outputId="f0538170-d8d9-4d55-df00-569044bc69b1"
len(X_train), len(X_val), len(X_test)

# %% colab={"base_uri": "https://localhost:8080/"} id="ck22nlTxivwz" outputId="520d8fe7-68e5-4142-e852-b4789c6760ec"
mlb = MultiLabelBinarizer()
mlb.fit(data["genre"].values)

# %% colab={"base_uri": "https://localhost:8080/"} id="jpaPvC8xsWia" outputId="36007521-33ac-4c7f-c4be-a5825423d6c8"
len(mlb.classes_)

# %% id="D20UnS_Zchxu"
LABELS = mlb.classes_

# %% id="17uXQzUSlS3g"
y_train = mlb.transform(y_train)
y_val = mlb.transform(y_val)
y_test = mlb.transform(y_test)


# %% [markdown] id="IlukJkRH017-"
# ### **Almacenamiento**
# - Toda la data descargada la almacenamos en TFRecords, dado que es un formato liviano y muy rápido cuando de entrenamientos se habla.

# %% id="pFappwUbl6nF"
def process_image(img_file: str) -> ByteString:
    """
    Process image to byte buffer.

    :param img_file: The file to be buffered
    :return: The image buffered
    """
    with open(img_file, "rb") as f:
        image_buffer = f.read()
    return image_buffer


def int64_feature(value: Union[List[int], int]) -> tf.train.Feature:
    """
    Transform integer
    to Tensorflow Feature list.

    :param value: The value to be transformed
    :return: The value transformed to TF Feature
    """
    if not isinstance(value, list):
        value = [value]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value: ByteString) -> tf.train.Feature:
    """
    Transform a byte string to a
    Tensorflow Feature list.

    :param value: The byte string to be transformed
    :return: The value transformed to TF Feature
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_example(
    image_buffer: ByteString,
    label: int,
) -> tf.train.Example:
    """
    Transform the image buffer, the
    label and the bounding box to an Example,
    which is going to be serialized in
    the TFRecord file.

    :param image_buffer: The image buffer string
    :param label: The label of the image
    :param bbox: The array of coordinates
    of the location of the class bird
    :return: The TF Example object ready to be
    Serialized in the TFRecord File
    """
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image": bytes_feature(tf.compat.as_bytes(image_buffer)),
                "class": int64_feature(label),
            }
        )
    )
    return example


def create_tfrecord(
    file_collection: List[str],
    label_collection: List[int],
    tfrecord_name: str = "collection.tfrec",
) -> None:
    """
    Serialize into TFRecord file all the Tensorflow
    examples created in the file collection.

    :param file_collection: The collection of images,
    labels and bboxes to be stored
    :param tfrecord_name: The name of the TFRecord
    file to be created
    """
    with tf.io.TFRecordWriter(tfrecord_name) as writer:
        for image_filepath, label_array in tqdm(list(zip(file_collection, label_collection))):
            label_array = [int(label) for label in label_array]
            image_buffer = process_image(image_filepath)
            example = convert_to_example(image_buffer, label_array)
            writer.write(example.SerializeToString())


# %% colab={"base_uri": "https://localhost:8080/", "height": 113, "referenced_widgets": ["f186269d1e7d4bb5aacb17ed26703ae2", "56f4001fb52c4e7496a68039b127f51f", "212252bc016c494ca18fe4e6a46d00c0", "5be5dadd392a4429889a3367fa684b4b", "813ad53f09f1467aa61d7afce992d66b", "637998300b3942899bb4efbb648a3d7a", "9ce1a0e62f744338a5a2db286b7f2046", "fca6266b6ffa476ea25aa36128041f33", "a87d5587e3f74b70b278a535efc221da", "67f42f328bd840c6893f2c977e5f4cbf", "5173e9533e3a40d591c69f003f09b617", "95f105aacd4742b59bedc9f62329a33d", "099bb6a5c6654a0f869a4d9046dd3572", "e826dc4167ee4861a37c3c7ce0665dfe", "6e489442e730418e9bc652b2db268ca7", "c6b50e6f40114e1280bdc0f30401471c", "2ef902cb1e1645faaacfb4e7bd449b44", "69e41eab5ad14ded896ce50e4311f73a", "e67b931a28904744ae7b49bc73916bb2", "ee2696cce2af49de8f61597b5cc70b00", "50ba6a04214d4542809f4a593710a7ac", "c2815cc61b8d4eb3bf3a4e2be33862fc", "73265bb3a38d4e9fbbea000d584d7e19", "3dd9983691e84f149621abb5946563e3", "9a5e29a6d72c44459defc9b9bead96ff", "7f3a983b73914f5f966dfc4c54c54004", "098d1f56628f4867b1bc0c37c489763d", "188bb0fa8cbc4701b16579c493a1d52e", "13be7d7b4ddc4f33912b40d7beed064d", "1087f0676ad7477487bc4de868a47c4d", "1ccf2b1d57844c238f3e5c605cd7c252", "54fc3797476240eab2b9a54347999641", "cbfddbd971c64da28959069d46fdedd8"]} id="RyPWTyH-nPZA" outputId="dce06196-9245-4a66-ddc7-52c67351fa2d"
create_tfrecord(X_train, y_train, "train.tfrec")
create_tfrecord(X_val, y_val, "val.tfrec")
create_tfrecord(X_test, y_test, "test.tfrec")

# %% [markdown] id="fLuKSGJaIKzc"
# ## **Modelado**
# ### **Descarga de los Datos Procesados**

# %% colab={"base_uri": "https://localhost:8080/"} id="niWn09FTrulP" outputId="af719f7b-7f60-48d8-aa7d-b3a49b8a40e3"
# %%shell
# mkdir data
# echo "Descargando data..."
# echo "Entrenamiento"
gdown -q 1i9SA3FN1p6iax44BmVUrttRBh7oHH5pl -O data/train.tfrec
# echo "Validación"
gdown -q 1dYFyFpWxSmkSrxkWho2KgR-jpEBP8l-m -O data/val.tfrec
# echo "Testing"
gdown -q 1aASAkmHTemnwPERIH94taQ1Fr9XqXgK0 -O data/test.tfrec

# %% id="ms_WfFjY52C7"
TRAIN_SIZE, VAL_SIZE, TEST_SIZE = (23311, 5828, 7285)
BATCH_SIZE = 64
NUM_CLASSES = 18

DATA_SHAPE = (128, 128)


# %% [markdown] id="X1gTB1qZ1QoD"
# ### **Procesamiento y Carga**

# %% id="ww0_pBqFMKfF"
def parse_image_function(
    example_input: tf.Tensor,
) -> Tuple[tf.Tensor]:
    """
    Parse a TFRecord serialized single example.

    :param example_input: The serialized TF
    Example to be parsed
    :return: The resulting image and label of
    the parsed Example
    """
    # We extract the feature description from the example
    # represented by features of fixed length and fixed type
    image_feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "class": tf.io.FixedLenFeature([18], tf.int64),
    }

    # We parse example by example
    feature = tf.io.parse_single_example(example_input, image_feature_description)

    # We get the features from the feature dict
    image = feature["image"]
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0

    label = tf.cast(feature["class"], tf.int32)

    return image, label


def process_image(
    image: tf.Tensor,
    label: tf.Tensor,
) -> Tuple[tf.Tensor]:
    """
    Process the image, label and bbox.

    :param image: The image to be processed
    :param label: The label of the image
    :param bbox: The bbox of the image
    :return: The image, label and bbox processed
    """
    shape = tf.shape(image)

    image = tf.image.resize(image, DATA_SHAPE)
    # image = tf.image.rgb_to_grayscale(image)

    return image, label


def augment_data(
    img: tf.Tensor,
    label: tf.Tensor,
) -> Tuple[tf.Tensor]:
    """
    Augment data from image functions.

    :param img: The image to be processed
    :param label: The label of the image
    :return: The image processed and the label
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.3)

    return img, label


def performance(
    dataset: tf.data.Dataset,
    train: bool = True,
) -> tf.data.Dataset:
    """
    Function to boost dataset load performance.

    :param dataset: The dataset to be boosted
    :param train: Flag to indicate the nature of the dataset
    :return: The dataset boosted
    """
    if train:
        # Shuffle the dataset to a fixed buffer sample
        dataset = dataset.shuffle(512, reshuffle_each_iteration=True)
        # The number of batches that will be parallel processed
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    # Repeat the incidences in the dataset
    dataset = dataset.repeat()
    # Create batches from dataset
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


# %% id="DToiTd_TrfSc"
train_ds = tf.data.TFRecordDataset("data/train.tfrec")
val_ds = tf.data.TFRecordDataset("data/val.tfrec")
test_ds = tf.data.TFRecordDataset("data/test.tfrec")

# %% id="ksVfr4G_o5JH"
train_ds = train_ds.map(parse_image_function)
val_ds = val_ds.map(parse_image_function)
test_ds = test_ds.map(parse_image_function)

# %% id="k-V7qT_NrpMt"
train_ds = train_ds.map(process_image)
val_ds = val_ds.map(process_image)
test_ds = test_ds.map(process_image)

# %% id="YZERstL0r34G"
train_ds = train_ds.map(augment_data)

# %% colab={"base_uri": "https://localhost:8080/"} id="Kup1nSTe8R5f" outputId="2094efb9-0292-4de2-de96-ef46fc8d9aa9"
for image, label in train_ds.take(5):
    tf.print("Image shape: ", image.shape)
    tf.print("Label shape: ", label.shape)
    tf.print("Label: ", label)

# %% id="ULQX9ovm8UPB"
train_ds = performance(train_ds)
val_ds = performance(val_ds, False)
test_ds = performance(test_ds, False)

# %% colab={"base_uri": "https://localhost:8080/"} id="7UIMkRWJ8c01" outputId="4093636a-18e6-41f5-876f-6c7abae95841"
for img_batch, label_batch in train_ds.take(1):
    tf.print(img_batch.shape)
    tf.print(label_batch.shape)


# %% [markdown] id="BCE5uGke1i61"
# ### **Creación de Métrica y Función de Pérdida**
# - En este caso usamos una métrica global, la F1. Esta la necesitamos puesto que nos ayuda a ponderar de forma global el resultado con respecto a los verdaderos postivos, falsos positivos y falsos negativos. La F1 es útil en este caso porque tenemos una predicción para cada etiqueta, en lugar de una única predicción.

# %% id="72dxEWHaO47e"
class MacroSoftF1(keras.losses.Loss):
    def __init__(
        self,
        name: str = "MacroSoftF1",
        epsilon: float = 1e-16,
        **kwargs,
    ) -> None:
        """
        Initializer function.

        :param name: The name of the loss function
        :param epsilon: The epsilon softener
        """
        super(MacroSoftF1, self).__init__(name=name, **kwargs)
        self.epsilon = epsilon

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """
        Loss function caller.

        :param y_true: The true labels
        :param y_pred: The predicted labels
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # True positives
        tp = tf.reduce_sum(y_pred * y_true, axis=0)
        # False positives
        fp = tf.reduce_sum(y_pred * (1 - y_true), axis=0)
        # False negatives
        fn = tf.reduce_sum((1 - y_pred) * y_true, axis=0)
        # Soft f1 calculated
        soft_f1 = 2 * tp / (2 * tp + fn + fp + self.epsilon)
        cost = 1 - soft_f1
        macro_cost = tf.reduce_mean(cost)
        return macro_cost


class MacroF1(keras.metrics.Metric):
    def __init__(
        self,
        name: str = "MacroF1",
        thresh: float = 0.5,
        epsilon: float = 1e-16,
        **kwargs,
    ) -> None:
        """
        Initializer function.

        :param name: The name of the metric
        :param thresh: The threshold of the scores
        :param epsilon: The epsilon softener
        """
        super(MacroF1, self).__init__(name=name, **kwargs)
        self.batch_macro_f1 = self.add_weight(
            name="macro_f1",
            initializer="zeros",
        )
        self.count = self.add_weight(
            name="count",
            initializer="zeros",
        )
        self.thresh = thresh
        self.epsilon = epsilon

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """
        Update the metric weights.

        :param y_true: The true labels
        :param y_pred: The predicted labels
        :param sample_weight: Optional weighting of each example
        """
        y_pred_temp = tf.cast(tf.greater(y_pred, self.thresh), tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        tp = tf.cast(tf.math.count_nonzero(y_pred_temp * y_true, axis=0), tf.float32)
        fp = tf.cast(
            tf.math.count_nonzero(y_pred_temp * (1 - y_true), axis=0),
            tf.float32,
        )
        fn = tf.cast(
            tf.math.count_nonzero((1 - y_pred_temp) * y_true, axis=0),
            tf.float32,
        )
        f1 = 2 * tp / (2 * tp + fn + fp + self.epsilon)
        macro_f1 = tf.reduce_mean(f1)
        self.batch_macro_f1.assign_add(macro_f1)
        self.count.assign_add(1)

    def result(self) -> tf.Tensor:
        """
        Calculate the result of weights.
        """
        return self.batch_macro_f1 / self.count


# %% id="5QZVOcy3MKtv"
Dense = functools.partial(keras.layers.Dense, activation="relu")

# %% [markdown] id="0G2JvlLV435f"
# - La arquitectura utilizada es el Perceptrón Multicapa, por lo que, necesitamos convertir nuestras imágenes a vectores.
# - En la capa final usamos una capa densa con 18 neuronas, cada una asociada a cada etiqueta, y una activación sigmoide. La activación sigmoide se hace porque buscamos una predicción entre 0 y 1 para cada género de película, dado que estamos en un contexto multietiqueta, en lugar de que la sumatoria de todas las predicciones sea igual a 1, lo que nos indica que habrá un único género que tendrá predominancia sobre el resto.

# %% id="tQlsV982K3YV"
model = keras.Sequential(
    [
        keras.layers.Input(shape=(*DATA_SHAPE, 3)),
        keras.layers.Flatten(),
        Dense(1024),
        Dense(512),
        Dense(256),
        Dense(128),
        Dense(64),
        Dense(32),
        keras.layers.Dense(NUM_CLASSES, activation="sigmoid"),
    ]
)

# %% id="CF5MbxfYMt1V"
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    metrics=[
        tfa.metrics.F1Score(
            num_classes=NUM_CLASSES,
            average="macro",
            threshold=0.5,
        )
    ],
    loss=MacroSoftF1(),
)

# %% id="yC-iJ7D5M88o"
params = {
    "steps_per_epoch": TRAIN_SIZE // BATCH_SIZE,
    "validation_steps": VAL_SIZE // BATCH_SIZE,
    "epochs": 5,
    "validation_data": val_ds,
}

# %% colab={"base_uri": "https://localhost:8080/"} id="WVU6S81DM_xm" outputId="1f49119c-7f61-4abd-bd81-6408cabdcde9"
model.fit(train_ds, **params)

# %% colab={"base_uri": "https://localhost:8080/"} id="uPjXvcDaNCQb" outputId="2937802e-d054-4a53-ffe6-d298858e3bd3"
model.evaluate(test_ds, steps=TEST_SIZE // BATCH_SIZE)


# %% [markdown] id="6IycUIts11_Q"
# ## **Optimización de Hiperparámetros**
# - En esta ocasión vamos a tunear la función de activación y el inicializador del kernel.

# %% id="_1jdETppaIH6"
def model_builder(hyperparameters: kt.HyperParameters) -> keras.Model:
    """
    Build model for hyperparameter tuning.

    :param hyperparameters: The hyperparameter container
    for the hyperparameter space
    :return: The model with the hyperparameters
    """
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(*DATA_SHAPE, 3)))
    model.add(keras.layers.Flatten())
    for units in range(10, 4, -1):
        model.add(
            keras.layers.Dense(
                2 ** units,
                activation=hyperparameters.Choice(
                    "activation",
                    ["relu", "softplus", "swish"],
                ),
                kernel_initializer=hyperparameters.Choice(
                    "kernel_initializer",
                    ["glorot_normal", "he_normal"],
                ),
            ),
        )
    model.add(keras.layers.Dense(NUM_CLASSES, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        metrics=[
            tfa.metrics.F1Score(
                num_classes=NUM_CLASSES,
                average="macro",
                threshold=0.5,
            )
        ],
        loss=MacroSoftF1(),
    )
    return model


# %% id="1Nxv5YUZMuk8"
tuner = kt.RandomSearch(
    model_builder,
    objective=kt.Objective("val_f1_score", direction="max"),
    seed=1234,
    max_trials=10,
)

# %% id="x7On-tG-NUbo"
params["epochs"] = 10

# %% colab={"base_uri": "https://localhost:8080/"} id="AaXgV8JzNH-h" outputId="43efe613-06d4-4fa8-934e-bc1da0a1cea4"
with tf.device("/device:GPU:0"):
    tuner.search(
        train_ds,
        **params,
    )

# %% id="8xvu_oEPNLUQ"
best_model = tuner.get_best_models()[0]

# %% colab={"base_uri": "https://localhost:8080/"} id="LGZ61_iJNNO_" outputId="89f3f80f-2e1e-4435-fd6a-da4cac11df34"
best_model.summary()

# %% colab={"base_uri": "https://localhost:8080/", "height": 976} id="nfnvQIymcqNj" outputId="2b75e585-5c48-4f64-a4d7-6a9801f776a5"
keras.utils.plot_model(
    best_model,
    show_layer_activations=True,
    show_shapes=True,
)

# %% id="8r-FsTwGsEWv"
params["epochs"] = 50

# %% [markdown] id="g-6mcEhA5KxH"
# - Hacemos un pequeño reentrenamiento de nuestro mejor modelo para ver cuáles son los mejores resultados que se pueden obtener.

# %% colab={"base_uri": "https://localhost:8080/"} id="HvQzqUTusC4E" outputId="b9d378f8-ad7e-4db5-ad9c-65c0d5000c26"
best_model.fit(train_ds, **params)

# %% [markdown] id="G9DnDQ512Hwy"
# ## **Validación**
# - una vez hecho el entrenamiento procedemos a validar nuestros resultados contra lo que tenemos en testing.

# %% colab={"base_uri": "https://localhost:8080/"} id="hhkISjWOc45d" outputId="202d6789-7fe8-4308-e6d2-b1c0c46e278a"
best_model.evaluate(test_ds, steps=TEST_SIZE // BATCH_SIZE)

# %% id="vE5d6rk8dO6M"
for X_batch, y_batch in test_ds.take(1):
    ...

# %% id="VcoSmFf4du1k"
y_batch_pred = best_model.predict(X_batch)

# %% id="h9Kp-ZYVdWzz"
X_batch = X_batch.numpy()
y_batch = y_batch.numpy()

# %% id="F89eOHa4eg4Q"
array_to_label = lambda x: LABELS[x]

# %% id="sD5ALt5jeTm2"
indexes = np.random.randint(0, 64, size=9)

X = X_batch[indexes]
y_true = list(
    map(
        extract_label,
        list(map(array_to_label, y_batch[indexes].astype(bool))),
    ),
)
y_pred = list(
    map(
        extract_label,
        list(map(array_to_label, y_batch_pred[indexes] > 0.8)),
    ),
)
titles = [f"True: {t} \n Pred: {p}" for t, p in list(zip(y_true, y_pred))]

# %% colab={"base_uri": "https://localhost:8080/", "height": 729} id="6YJuVYqwgSWb" outputId="49d24b9c-5658-4d42-d6f3-f53f9ffe5a72"
gen.visualize_subplot(
    imgs=X,
    titles=titles,
    division=(3, 3),
    figsize=(20, 10),
)
plt.tight_layout()

# %% colab={"base_uri": "https://localhost:8080/"} id="Y0nAD3Wg2Qd3" outputId="3c0cc3e9-af63-4088-9086-2d4713429984"
print(classification_report(y_batch, y_batch_pred > 0.8))

# %% [markdown] id="dlHBQzWs5cLm"
# ## **Conclusiones**
# - El modelo tuvo un rendimiento bajo puesto que es un perceptrón que trabaja con imágenes. Por lo tanto, al convertirlas en vectores perdemos la percepción espacial de cada imagen y eso no ayuda a que las predicciones sean de buena calidad. Por otra parte, si se hiciera uso de una función descriptora de imágen, daría mejores resultados al final.
# - De cara al negocio, vemos que nuestro modelo logra predecir varios géneros por cada póster de película, aunque lo hace con poca precisión. Para un futuro proyecto, sería ideal probar arquitecturas neuronales más orientadas al análisis de imágen, como la red neuronal convolucional.

# %% id="TVmfWHNb2UGH"
