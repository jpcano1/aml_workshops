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

# %% [markdown] id="rft1MP5eCPav" pycharm={"name": "#%% md\n"}
# # **Análisis con Machine Learning**
# ## **Taller 1**
# #### **Andrea Bayona - Juan Pablo Cano**
#
#
# En la actualidad, el sector inmobiliario ruso está en pleno auge. Ofrece muchas oportunidades emocionantes y un alto rendimiento en cuanto a estilos de vida e inversiones. El mercado inmobiliario lleva varios años en fase de crecimiento, lo que significa que todavía se pueden encontrar propiedades a precios muy atractivos, pero es muy probable que aumenten en el futuro. Para poder entender el mercado, una inmobiliaria rusa le ha brindado la información de la venta de más de 45 mil inmuebles entre los años de 2018 y 2021. Y quieren entender cuáles son las características principales que inciden en los precios de venta, para poder proponer planes de construcción de inmuebles en las áreas urbanas disponibles, que tomen en cuenta estas características.

# %% colab={"base_uri": "https://localhost:8080/", "height": 67, "referenced_widgets": ["f36e5fc2011e4610ab7023be86219b84", "c415eeb92f7043e8b96a57a0b2f66de0", "573acf786cbc45c99c7929bf26ac3803", "0fbc465aa99244fb826cd8c2d708a2b0", "a5ce9451dee94807a0167712ab0ea83e", "787091749c8c4cc6b4e1e1abc81a99c9", "d520ac2d610c442fbd45003ad6c49e45", "1db1e8f04e58403eb3961e8e2a0d2848", "6fcd4db153e6466795b5d7d9ca9fed0c", "817bda5a33ca4889ae21d17274e36833", "78e0622191554a49a350cbe47d5fbc50"]} id="Tb2Jg3f9jx-S" outputId="ce24700f-f124-47de-9608-bd6ac01065de" pycharm={"name": "#%%\n"}
# !shred -u setup_colab_general.py
# !wget -q "https://github.com/jpcano1/python_utils/raw/main/setup_colab_general.py" -O setup_colab_general.py
import setup_colab_general as setup_general

setup_general.setup_general()

# %% id="yqAdW4WNlYXN" pycharm={"name": "#%%\n"}
# !pip install --disable-pip-version-check --progress-bar off -q https://github.com/pandas-profiling/pandas-profiling/archive/master.zip
# !pip install --disable-pip-version-check --progress-bar off -q tabulate

# %% [markdown] id="Am5V_Yb1ntVC" pycharm={"name": "#%% md\n"}
# ## **Importando la librerías necesarias**
# %% id="ebTvp-cCwAsQ" pycharm={"name": "#%%\n"}
import os

import matplotlib.pyplot as plt

plt.style.use("seaborn-deep")

# Librerías extras
import itertools
from typing import Optional

import mlflow
import numpy as np
import pandas as pd

# import pandas_profiling
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tabulate import tabulate
from utils import general as gen

# %% colab={"base_uri": "https://localhost:8080/", "referenced_widgets": ["8f32ac9affd94093a82965446c22a1d6", "4b1d20d0b9e74ee89d8ef7268f013fe8", "d9286e6079d14ad88f1a39687feeea4b", "7532de9b083d4a978579b7e9d5d01f24", "18a0f8191bf0443e8a41e252a4fa9a4a", "1773c0cda1864c4884dcbaef0bfab8db", "489c0dea7aca4a038221829119dd6b2c", "72647a8834254e5084010cdd80b9feec", "d6b1b23f40124546884d41b134d3bc8b", "e96c2b8830294aa9a4cda6782992a319", "e31accab95874bca9428ad25996f8dbe"]} id="CtzhV51Hwa5p" outputId="f479c4a9-fe08-458e-dc21-2cd5c9e10a96" pycharm={"name": "#%%\n"}
data_url = (
    "https://raw.githubusercontent.com/"
    "Camilojaravila/202210_MINE-4206_ANAL"
    "ISIS_CON_MACHINE_LEARNING/main/Taller%"
    "201/russian_prices.csv"
)
gen.download_content(data_url, filename="russian_prices.csv")

# %% [markdown] id="IuYNGV_wnxjN" pycharm={"name": "#%% md\n"}
# ## **Lectura y perfilamiento**
#
# ### **Diccionario de Datos**
# La inmobiliaria ha construido el siguiente diccionario de datos:
#
# * date - Fecha de publicación del anuncio.
# * time - Tiempo que la publicación estuvo activo.
# * geo_lat - Latitud.
# * geo_lon - Longitud.
# * region - Region de Rusia. Hay 85 regiones en total.
# * building_type - Tipo de Fachada. 0 - Other. 1 - Panel. 2 - Monolithic. 3 - * Brick. 4 - Blocky. 5 - Wooden.
# * object_type - Tipo de Apartmento. 1 - Secondary real estate market; 2 - New * building.
# * level - Piso del Apartamento.
# * levels - Número de pisos.
# * rooms - Número de Habitaciones. Si el valor es "-1", Significa que es un "studio apartment".
# * area - Área total del apartamento (m2).
# * kitchen_area - Área de la Cocina (m2).
# * price - Precio. En rublos

# %% [markdown] id="tvVbLTlBE6mh" pycharm={"name": "#%% md\n"}
# A continuación, se leen los datos y se revisan las primeras líneas para verficar que la carga fue exitosa

# %% id="kj0nEuMvxWad" pycharm={"name": "#%%\n"}
russian_prices_df = pd.read_csv("data/russian_prices.csv")

# %% colab={"base_uri": "https://localhost:8080/"} id="2wS1sFwMj6uO" outputId="1a289b70-6d2f-43ff-c40f-b98ecaba21d7" pycharm={"name": "#%%\n"}
russian_prices_df.head()

# %% colab={"base_uri": "https://localhost:8080/"} id="wfg_2vpIj8-_" outputId="61788701-a72e-4bbe-b3d6-1e872a8e9268" pycharm={"name": "#%%\n"}
russian_prices_df.info()

# %% id="Wl1BKpS4xAw2" pycharm={"name": "#%%\n"}
profiler = pandas_profiling.ProfileReport(russian_prices_df, dark_mode=True)

# %% [markdown] id="zfOPUCD0wNxc" pycharm={"name": "#%% md\n"}
# - El perfilamiento se encuentra en los anexos.

# %% colab={"base_uri": "https://localhost:8080/", "referenced_widgets": ["43552aa81edb4d549039620ba4f5e219", "59b1f7c7890044eeabc09b440ec2179b", "658be1c5915748eaa1e145ee16025c68", "cb1a99b1c43047e4b18dbe88cfce10bb", "5528a149e1da4d76ad182fb73fb7d540", "8bac1b90e20549ccb5954d64b329229e", "ff378d55cf9d471abe1a70f9fae5a2a4", "5de5caaef3224713b8baff7e03b672e7", "f532373581a14500a9a52b55643d0cd2", "c044d2abff9e4234b24282312d63aabb", "497800d0848f40478c2385c362944b70", "e9ac7ae4f4e148d49189cf2e82e10b85", "1b86aad8a58d4855850c7da938397545", "a84574e678d54475adccd2f1830d98b7", "6fba2332a3904402bc5849ca2ec8d8b4", "c7f19e79908a4580b9c21802660c16eb", "3f000da7a9554c85b81577680406a5fc", "035209bb241840ecba66b532383e2fe3", "ff6d580988ea484d88dc3652b597200b", "d7b21b39409e4a9896e8b9c89ee719df", "3595af6e985442b2b03a99eb45e392a6", "1aa66f2d218c46f1b1977bdd4a06a29f", "62da8655840a48588d5f17c851a7704e", "9a019b58dde1427ba137691ce5a9ca45", "fcb628673ba54f3aa18bac9e19ae479d", "6b1d0ddca02b4a5080e3a14f09617876", "43f62b6498844699824740203cbad04e", "29389e45904e4f659450c955be8e340a", "aa6e881d2ce446be9e270244ba2ad91c", "c3a3d6c6bf954f8bbfbb4280e2c60474", "83ffe96d67304724be8a594b3c78eaa5", "ceba62c299b34d2088bc8e0f7e2ef3b6", "92d0c71528b44bff96eccedd65a8f1b0", "8167ec02a4c548ffade77e4532441af5", "db235209d453453aa12a51c2a3cdec89", "8b896ac119904b918f4dc458b111a726", "5655bc92cc4b4d14a0a94092217beb32", "f9157981d403417ebcba1ee5551479cc", "a33e61b591394ddfb951523aa84a2e52", "6396e9cb24de43598929c0e0fc8c620a", "e431c4fce170458f87f822281db1e3d6", "b7cee479ea9e45e181ff7997e2cd2ca8", "1dced353eb414a9392a77d9c41c1de95", "b9c729011d924b4699445652699f8503"]} id="ztSi0cuPxffs" outputId="0213797c-0761-4293-c541-e30ca32dfa90" pycharm={"name": "#%%\n"}
if not os.path.exists("profiling_reports"):
    os.makedirs("profiling_reports")
profiler.to_file("profiling_reports/russian_prices_profile.html")

# %% [markdown] id="oO11RRE2VWAN" pycharm={"name": "#%% md\n"}
# - Las siguientes columnas se eliminaron bajo el supuesto de que no son necesarias para el objetivo de negocio. La primera columna es un identificador de propiedad, ergo, no es significativa. Las columnas `time` y `date` son columnas relacionadas a la publicación, más no a la propiedad perse, por lo tanto, no son significativas para nuestro modelo.

# %% id="xUQji5iQi4oY" pycharm={"name": "#%%\n"}
columns_to_delete = [
    "Unnamed: 0",
    "time",
    "date",
]

# %% id="HuVpJcPpjox-" pycharm={"name": "#%%\n"}
russian_prices_df.drop(columns=columns_to_delete, inplace=True)

# %% [markdown] id="k9tWqKeNsACK" pycharm={"name": "#%% md\n"}
# - Todas las columnas con valores nulos fueron removidas

# %% id="rdgm3ridkZhS" pycharm={"name": "#%%\n"}
russian_prices_df.dropna(inplace=True)

# %% colab={"base_uri": "https://localhost:8080/"} id="2fvSWw_hj1wS" outputId="aea14985-ac08-4ddb-c820-f8e4e371ecd8" pycharm={"name": "#%%\n"}
russian_prices_df.info()

# %% id="NXiS5wFokqHi" pycharm={"name": "#%%\n"}
russian_prices_df = russian_prices_df.apply(lambda x: x.astype("int32"))
russian_prices_df["object_type"] = russian_prices_df["object_type"].apply(
    lambda x: 2 if x == 11 else x
)
russian_prices_df["rooms"] = russian_prices_df["rooms"].apply(lambda x: -1 if x == -2 else x)

# %% id="dYivdCPnwfuo" pycharm={"name": "#%%\n"}
rows_to_drop = russian_prices_df.query(
    "kitchen_area + 5 >= area | area <= 10 | price <= 2000"
).index
russian_prices_df = russian_prices_df.drop(rows_to_drop).reset_index(drop=True)

# %% id="V7BDrFLJmcN9" pycharm={"name": "#%%\n"}
X, y = russian_prices_df.drop("price", axis=1), russian_prices_df["price"]

# %% id="Gh_LIlQTmm0S" pycharm={"name": "#%%\n"}
full_X_train, X_test, full_y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)
X_train, X_val, y_train, y_val = train_test_split(
    full_X_train, full_y_train, test_size=0.2, random_state=1234
)

# %% colab={"base_uri": "https://localhost:8080/"} id="eEtwMvsAmogC" outputId="4362362f-68c0-4c00-d671-495085c99382" pycharm={"name": "#%%\n"}
X_train.shape, y_train.shape

# %% colab={"base_uri": "https://localhost:8080/"} id="M3JwOPusnIpn" outputId="66c0f45e-ed9e-4e7a-bc3e-ab59c717bc4a" pycharm={"name": "#%%\n"}
X_val.shape, y_val.shape

# %% colab={"base_uri": "https://localhost:8080/"} id="0QveAb5em00v" outputId="c75a4fc7-79a8-4c3f-b185-8b02830bd599" pycharm={"name": "#%%\n"}
X_test.shape, y_test.shape


# %% [markdown] id="rP-CN-yeoJ_9" pycharm={"name": "#%% md\n"}
# ## **Modelamiento**

# %% [markdown] id="16UKiXe_TM12" pycharm={"name": "#%% md\n"}
# ### **Regresión Polinómial**
# #### **Entrenamiento (Sin estandarización)**

# %% [markdown] id="zHyXQXfb2qAG" pycharm={"name": "#%% md\n"}
# Se define la clase para realizar la transformación polinomial de nuetras variables

# %% id="1EeXFbK_pP7a" pycharm={"name": "#%%\n"}
class ToPolynomial(BaseEstimator, TransformerMixin):
    def __init__(self, k: int = 2) -> None:
        self.k = k

    def fit(self, X, y):
        return self

    def transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        columns = X.columns
        X_train_pol = pd.concat(
            [X ** (i + 1) for i in range(self.k)], axis=1
        )  # Polinomios sin interacciones
        X_train_pol.columns = np.reshape(
            [[i + " " + str(j + 1) for i in columns] for j in range(self.k)], -1
        )
        temp = pd.concat(
            [X[i[0]] * X[i[1]] for i in list(itertools.combinations(columns, 2))], axis=1
        )  # Combinaciones sólo de grado 1
        temp.columns = [" ".join(i) for i in list(itertools.combinations(columns, 2))]
        X_train_pol = pd.concat([X_train_pol, temp], axis=1)
        return X_train_pol


# %% [markdown] id="Fj3hG4RM2wLO" pycharm={"name": "#%% md\n"}
# Se crea un pipeline para encapsular los pasos de entrenamiento de nuestro modelo. Primero se realiza la transformación polinamial de las variables y estas se utilizan para entrenar el modelo de regresión lineal.

# %% colab={"base_uri": "https://localhost:8080/"} id="2yMSJKDloOZi" outputId="de3292b9-2b6c-42ee-ebb0-e0af96d5e87d" pycharm={"name": "#%%\n"}
estimators = [("polinomial", ToPolynomial()), ("regresion", LinearRegression())]

pipe_pol = Pipeline(estimators)

pipe_pol.fit(X_train, y_train)

# %% [markdown] id="0G2Ruy0T4DUD" pycharm={"name": "#%% md\n"}
#  Parámetros entrenados por la Regresión Polinomial

# %% colab={"base_uri": "https://localhost:8080/"} id="eCPY09nsTcWK" outputId="2ca0f684-34c3-482b-c1ed-6dcc3ac6f9b2" pycharm={"name": "#%%\n"}
reg_lineal = pipe_pol["regresion"]

print("Intercept: ", reg_lineal.intercept_)
print("Coefficients: ", reg_lineal.coef_)

# %% [markdown] id="epQPSIhSTWXo" pycharm={"name": "#%% md\n"}
# #### **Validación (Sin estandarización)**

# %% colab={"base_uri": "https://localhost:8080/"} id="-epy5ZCTT2hd" outputId="15884a59-7aca-42a3-dad3-1307f841f369" pycharm={"name": "#%%\n"}
y_pred = pipe_pol.predict(X_val)
y_pred

# %% colab={"base_uri": "https://localhost:8080/"} id="UzYpZljfT9qv" outputId="8971380f-21df-46e7-8982-391455700b5b" pycharm={"name": "#%%\n"}
r2_poly = r2_score(y_val, y_pred)
mse_poly = mean_squared_error(y_val, y_pred)
mae_poly = mean_absolute_error(y_val, y_pred)

print("------------ Polynomial Regression ------------")
print(f"R2-score: {r2_poly:.7f}")
print(f"Residual sum of squares (MSE): {mse_poly:.5f}")
print(f"Mean absolute error: {mae_poly:.5f}")


# %% [markdown] id="hWv4uZTyeZ1l" pycharm={"name": "#%% md\n"}
# #### **Comportamiento de los datos reales vs los datos predecidos**

# %% id="lt_fp3tjdVJy" pycharm={"name": "#%%\n"}
# %matplotlib inline


def draw_chart(y_val_p, y_pred_p, title, legend):
    fig, axs = plt.subplots(1, figsize=(20, 10))

    xvals = list(range(len(y_val_p[:50])))
    axs.plot(xvals, y_pred_p[:50], "bo-", label=legend)
    axs.plot(xvals, y_val_p[:50], "ro-", label="Real")

    axs.set(title=title, ylabel=y_train.name)
    axs.legend()

    plt.tight_layout()
    plt.show()


# %% colab={"base_uri": "https://localhost:8080/", "height": 729} id="HA1FneL3hcOX" outputId="6d31ad8a-ad46-415b-b9f1-0a04504241df" pycharm={"name": "#%%\n"}
draw_chart(y_val, y_pred, "Predicción con Regresión Polinomial", "Regresión Polinomial")

# %% [markdown] id="2L9PZ3OiPh9H" pycharm={"name": "#%% md\n"}
# #### **Entrenamiento (Con estandarización)**

# %% colab={"base_uri": "https://localhost:8080/"} id="WfoASDJqPhMt" outputId="365908bf-eec1-497a-f4db-ca22cc66fa37" pycharm={"name": "#%%\n"}
estimators_2 = [
    ("polinomial", ToPolynomial()),
    ("normalizar", StandardScaler()),
    ("regresion", LinearRegression()),
]

pipe_pol_s = Pipeline(estimators_2)

pipe_pol_s.fit(X_train, y_train)

# %% [markdown] id="H6VmO045P6MN" pycharm={"name": "#%% md\n"}
#  Parámetros entrenados por la Regresión Polinomial

# %% colab={"base_uri": "https://localhost:8080/"} id="h0C0dPUTQZfw" outputId="74a2f6ba-0d3d-462b-91f6-5d83be0ffea4" pycharm={"name": "#%%\n"}
reg_lineal_s = pipe_pol_s["regresion"]

print("Intercept: ", reg_lineal_s.intercept_)
print("Coefficients: ", reg_lineal_s.coef_)

# %% [markdown] id="xIYXvQSOQjqP" pycharm={"name": "#%% md\n"}
# #### **Validación (Con estandarización)**

# %% colab={"base_uri": "https://localhost:8080/"} id="w5I-DSlPQpO7" outputId="fce4b816-88c8-43d1-9d18-54fd6b318b64" pycharm={"name": "#%%\n"}
y_pred_1b = pipe_pol_s.predict(X_val)
y_pred_1b

# %% colab={"base_uri": "https://localhost:8080/"} id="DA3xqeyjQ1X9" outputId="489e69b3-10e1-4d03-9140-f580a5de423c" pycharm={"name": "#%%\n"}
r2_poly_s = r2_score(y_val, y_pred_1b)
mse_poly_s = mean_squared_error(y_val, y_pred_1b)
mae_poly_s = mean_absolute_error(y_val, y_pred_1b)

print("------------ Polynomial Regression ------------")
print(f"R2-score: {r2_poly_s:.7f}")
print(f"Residual sum of squares (MSE): {mse_poly_s:.5f}")
print(f"Mean absolute error: {mae_poly_s:.5f}")

# %% [markdown] id="0TwCRPfGRFve" pycharm={"name": "#%% md\n"}
# #### **Comportamiento de los datos reales vs los datos predecidos**

# %% colab={"base_uri": "https://localhost:8080/", "height": 729} id="9pWcfymsRGYG" outputId="6ef5214c-bcbf-430a-989d-41a036201963" pycharm={"name": "#%%\n"}
# %matplotlib inline
draw_chart(
    y_val,
    y_pred_1b,
    "Predicción con Regresión Polinomial (con estandarización)",
    "Regresión Polinomial",
)

# %% [markdown] id="ud_HEwjZoOrq" pycharm={"name": "#%% md\n"}
# ### **Regresión Ridge**
# #### **Entrenamiento (Sin estandarización)**
#
#

# %% colab={"base_uri": "https://localhost:8080/"} id="rnnb_rWBoRHa" outputId="538b5f4a-cac6-4919-babf-d9a095c15c05" pycharm={"name": "#%%\n"}
ridge_reg = Ridge()
ridge_reg.fit(X_train, y_train)

# %% colab={"base_uri": "https://localhost:8080/"} id="xe1WtO-IiepT" outputId="fdc0f7eb-78d2-4143-c21e-9e227f36c78a" pycharm={"name": "#%%\n"}
ridge_coef = dict(zip(X_train.columns, ridge_reg.coef_))
ridge_coef

# %% [markdown] id="Zdx6xzlXdN0b" pycharm={"name": "#%% md\n"}
# #### **Validación**

# %% id="26Wmz50fdRx5" pycharm={"name": "#%%\n"}
y_pred_2 = ridge_reg.predict(X_val)

# %% colab={"base_uri": "https://localhost:8080/"} id="EvQB0EHsi_Fm" outputId="03d2c814-640b-4820-c993-f5ef9ccbc785" pycharm={"name": "#%%\n"}
r2_ridge = r2_score(y_val, y_pred_2)
mse_ridge = mean_squared_error(y_val, y_pred_2)
mae_ridge = mean_absolute_error(y_val, y_pred_2)

print("------------ Ridge ------------")
print(f"R2-score: {r2_ridge:.7f}")
print(f"Residual sum of squares (MSE): {mse_ridge:.5f}")
print(f"Mean absolute error: {mae_ridge:.5f}")

# %% [markdown] id="UkIx8aTSe8JA" pycharm={"name": "#%% md\n"}
# #### **Comportamiento de los datos reales vs los datos predecidos**

# %% colab={"base_uri": "https://localhost:8080/", "height": 729} id="bkaeBDIuevsV" outputId="84b6d593-b37d-4859-b5fb-02403bb6967c" pycharm={"name": "#%%\n"}
# %matplotlib inline
draw_chart(y_val, y_pred_2, "Predicción con regresion Ridge", "Regresion Ridge")

# %% [markdown] id="9Pd-sS9eYGzD" pycharm={"name": "#%% md\n"}
# #### **Entrenamiento (Con estandarización)**

# %% colab={"base_uri": "https://localhost:8080/"} id="wcsl9CmhYJ_7" outputId="87b33712-c50b-48bd-a95e-c8c1f9a916d1" pycharm={"name": "#%%\n"}
pipeline_ridge = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("regressor", Ridge()),
    ],
)

pipeline_ridge.fit(X_train, y_train)

# %% colab={"base_uri": "https://localhost:8080/"} id="lLkPRJlSZZvS" outputId="cb398eb4-7f8d-45aa-f485-dc7b8ca62319" pycharm={"name": "#%%\n"}
ridge_coef = dict(zip(X_train.columns, pipeline_ridge.steps[1][1].coef_))
ridge_coef

# %% [markdown] id="dflufl4waDcF" pycharm={"name": "#%% md\n"}
# #### **Validación**

# %% id="6SYHTEbBaHiw" pycharm={"name": "#%%\n"}
y_pred_2b = pipeline_ridge.predict(X_val)

# %% colab={"base_uri": "https://localhost:8080/"} id="EelZTCKzaMSL" outputId="7728db79-cd20-4f04-f94a-60ddd0489289" pycharm={"name": "#%%\n"}
r2_ridge_s = r2_score(y_val, y_pred_2b)
mse_ridge_s = mean_squared_error(y_val, y_pred_2b)
mae_ridge_s = mean_absolute_error(y_val, y_pred_2b)

print("------------ Ridge (Con estandarización) ------------")
print(f"R2-score: {r2_ridge_s:.7f}")
print(f"Residual sum of squares (MSE): {mse_ridge_s:.5f}")
print(f"Mean absolute error: {mae_ridge_s:.5f}")

# %% [markdown] id="rbeZ9Bc2gxRu" pycharm={"name": "#%% md\n"}
# #### **Comportamiento de los datos reales vs los datos predecidos**

# %% colab={"base_uri": "https://localhost:8080/"} id="JRh9LRiNigzY" outputId="a1a96ea3-df1c-4366-da30-a625f913521f" pycharm={"name": "#%%\n"}
# %matplotlib inline
draw_chart(
    y_val, y_pred_2b, "Predicción con regresion Ridge (Con estandarización)", "Regresion Ridge"
)

# %% [markdown] id="mQ_FoF7RoRX9" pycharm={"name": "#%% md\n"}
# ### **Regresión Lasso**
# #### **Entrenamiento (Sin estandarización)**

# %% colab={"base_uri": "https://localhost:8080/"} id="TLdd00Wdm69O" outputId="e74fa362-6a4f-4f75-ceec-7ab3e051ca69" pycharm={"name": "#%%\n"}
lasso_reg = Lasso()
lasso_reg.fit(X_train, y_train)

# %% colab={"base_uri": "https://localhost:8080/"} id="atcLgUTkoaKF" outputId="60b2b2c5-dc4d-4db4-c7d6-0680047148ee" pycharm={"name": "#%%\n"}
lasso_coef = dict(zip(X_train.columns, lasso_reg.coef_))
lasso_coef

# %% [markdown] id="NE0rheOho8Oj" pycharm={"name": "#%% md\n"}
# #### **Validación**

# %% id="yRCL86NIokX6" pycharm={"name": "#%%\n"}
y_pred_3 = lasso_reg.predict(X_val)

# %% colab={"base_uri": "https://localhost:8080/"} id="htf4tAfFbPDT" outputId="10d2e563-2f80-45a6-b373-355d5a554da7" pycharm={"name": "#%%\n"}
r2_lasso = r2_score(y_val, y_pred_3)
mse_lasso = mean_squared_error(y_val, y_pred_3)
mae_lasso = mean_absolute_error(y_val, y_pred_3)

print("------------ Lasso ------------")
print(f"R2-score: {r2_lasso:.4f}")
print(f"Residual sum of squares (MSE): {mse_lasso:.5f}")
print(f"Mean absolute error: {mae_lasso:.5f}")

# %% [markdown] id="VMbo3Ep9kWGZ" pycharm={"name": "#%% md\n"}
# #### **Comportamiento de los datos reales vs los datos predecidos**

# %% colab={"base_uri": "https://localhost:8080/"} id="xeorxVPNkeMg" outputId="8741cd6e-692d-412a-fc08-699b4d269218" pycharm={"name": "#%%\n"}
# %matplotlib inline
draw_chart(y_val, y_pred_3, "Predicción con regresion Lasso", "Regresion Lasso")

# %% [markdown] id="FCjiiBzKcbTY" pycharm={"name": "#%% md\n"}
# #### **Entrenamiento (Con estandarización)**

# %% colab={"base_uri": "https://localhost:8080/"} id="Em51FsVrr62S" outputId="ce820441-1436-4d4a-ce70-f44d2003e55b" pycharm={"name": "#%%\n"}
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("regressor", Lasso()),
    ],
)

pipeline.fit(X_train, y_train)

# %% colab={"base_uri": "https://localhost:8080/"} id="QuqmS5LTdjr9" outputId="10de4b6b-69e5-4baf-a215-2c43c6a928dd" pycharm={"name": "#%%\n"}
lasso_coef = dict(zip(X_train.columns, pipeline.steps[1][1].coef_))
lasso_coef

# %% [markdown] id="TYpXgaMxdv77" pycharm={"name": "#%% md\n"}
# #### **Validación**

# %% id="TguKWWZwdn9a" pycharm={"name": "#%%\n"}
y_pred_3b = pipeline.predict(X_val)

# %% colab={"base_uri": "https://localhost:8080/"} id="Bv_znLZ-dxlq" outputId="6175e388-6104-4395-f6a7-4e9eb0e37922" pycharm={"name": "#%%\n"}
r2_lasso_s = r2_score(y_val, y_pred_3b)
mse_lasso_s = mean_squared_error(y_val, y_pred_3b)
mae_lasso_s = mean_absolute_error(y_val, y_pred_3b)

print("------------ Lasso (Con estandarización) ------------")
print(f"R2-score: {r2_lasso_s:.7f}")
print(f"Residual sum of squares (MSE): {mse_lasso_s:.5f}")
print(f"Mean absolute error: {mae_lasso_s:.5f}")

# %% [markdown] id="rA8zBvMtkYIp" pycharm={"name": "#%% md\n"}
# #### **Comportamiento de los datos reales vs los datos predecidos**

# %% colab={"base_uri": "https://localhost:8080/"} id="WvJeqiEAky85" outputId="b8a56fa3-2e1c-45c1-d04e-e87866aca8c4" pycharm={"name": "#%%\n"}
# %matplotlib inline
draw_chart(
    y_val, y_pred_3b, "Predicción con regresion Lasso (Con estandarización)", "Regresion Lasso"
)

# %% [markdown] id="8-9QFqjSlfFJ" pycharm={"name": "#%% md\n"}
# ### **Selección del mejor modelo**
# Tabla comparativa con los resultados de las métricas R2, MSE y MAE para los 3 modelos entrenados.

# %% colab={"base_uri": "https://localhost:8080/"} id="lWomqAnlmJK7" outputId="d14db8d7-2eff-489a-e2eb-4fd951555ff2" pycharm={"name": "#%%\n"}
info = {
    "Model": [
        "Poly Regression",
        "Poly Regression (con S)",
        "Ridge",
        "Ridge (con S)",
        "Lasso",
        "Lasso (con S)",
    ],
    "R2": [r2_poly, r2_poly_s, r2_ridge, r2_ridge_s, r2_lasso, r2_lasso_s],
    "MSE": [mse_poly, mse_poly_s, mse_ridge, mse_ridge_s, mse_ridge, mse_lasso_s],
    "MAE": [mae_poly, mae_poly_s, mae_ridge, mae_ridge_s, mae_lasso, mae_lasso_s],
}

print(tabulate(info, headers="keys", tablefmt="fancy_grid"))

# %% [markdown] id="ewVS7PkQd_1B" pycharm={"name": "#%% md\n"}
# ### **Optimización de hiperparámetros para el mejor modelo**

# %% id="eWSPKRNod7fJ" pycharm={"name": "#%%\n"}
parameters = {"polinomial__k": [2, 3, 4, 5], "normalizar": [StandardScaler(), "passthrough"]}

grid_search = GridSearchCV(
    pipe_pol_s, parameters, verbose=2, scoring="neg_mean_squared_error", cv=5, n_jobs=-1
)

# %% colab={"base_uri": "https://localhost:8080/"} id="PYd51X78SOoF" outputId="c15c344d-f0d1-4fbc-d199-c3b318ad5f8a" pycharm={"name": "#%%\n"}
grid_search.fit(X_train, y_train)

# %% colab={"base_uri": "https://localhost:8080/", "height": 711} id="GIPcgonDTMwX" outputId="8578c137-d2cc-4da6-eb96-c34b941823c6" pycharm={"name": "#%%\n"}
best_model = grid_search.best_estimator_

pd.DataFrame(grid_search.cv_results_)

# %% colab={"base_uri": "https://localhost:8080/"} id="w_kM-BRhTPcj" outputId="46490f76-64fb-4ceb-dfce-5891719a0aec" pycharm={"name": "#%%\n"}
grid_search.best_params_

# %% id="iS-M3gxsTgNX" pycharm={"name": "#%%\n"}
y_pred_final = best_model.predict(X_val)
y_pred_final_e = best_model.predict(X_train)

# %% colab={"base_uri": "https://localhost:8080/"} id="pxwDgaI4Tg4X" outputId="b1db8efc-3a22-4b57-cf6f-f202af0de1de" pycharm={"name": "#%%\n"}
r2_final_e = r2_score(y_train, y_pred_final_e)
mse_def_e = mean_squared_error(y_train, y_pred_final_e)
mae_poly_final_e = mean_absolute_error(y_train, y_pred_final_e)

print("------------ Polynomial Regression Entrenamiento------------")
print(f"R2-score: {r2_final_e:.7f}")
print(f"Residual sum of squares (MSE): {mse_def_e:.5f}")
print(f"Mean absolute error: {mae_poly_final_e:.5f}")

r2_final = r2_score(y_val, y_pred_final)
mse_def = mean_squared_error(y_val, y_pred_final)
mae_poly_final = mean_absolute_error(y_val, y_pred_final)

print("------------ Polynomial Regression Validacion ------------")
print(f"R2-score: {r2_final:.7f}")
print(f"Residual sum of squares (MSE): {mse_def:.5f}")
print(f"Mean absolute error: {mae_poly_final:.5f}")

# %% [markdown] id="O4adCe_SUJ7e" pycharm={"name": "#%% md\n"}
# #### **Comportamiento de los datos reales vs los datos predecidos**

# %% colab={"base_uri": "https://localhost:8080/", "height": 729} id="f9BUndGKUp_7" outputId="cfabc291-087a-494a-e612-c0e7319c2da6" pycharm={"name": "#%%\n"}
draw_chart(y_val, y_pred_final, "Predicción con Regresión Polinomial", "Regresión Polinomial")

# %% [markdown] id="ds0LmmujVqOr" pycharm={"name": "#%% md\n"}
# Variables del modelo

# %% colab={"base_uri": "https://localhost:8080/", "height": 441} id="lOpX332AVstA" outputId="ebbace87-82cc-4e3b-9440-f8d8c10d08b7" pycharm={"name": "#%%\n"}
reg_model = best_model["regresion"]
fake_df = best_model["polinomial"].transform(X_val)
print(f"Intercepto: {reg_model.intercept_}")
coef = list(
    zip(["Intercepto"] + list(fake_df.columns), [reg_model.intercept_] + list(reg_model.coef_))
)
coef = pd.DataFrame(coef, columns=["Variable", "Parámetro"])
coef

# %% colab={"base_uri": "https://localhost:8080/", "height": 423} id="EUvheoTsW2SN" outputId="3d3be266-8902-46ca-99a6-5b96133a7a82" pycharm={"name": "#%%\n"}
coef.sort_values("Parámetro")

# %% colab={"base_uri": "https://localhost:8080/", "height": 80} id="b0ErhLliWldS" outputId="d3274cd1-6d86-4569-d98a-20efc2405833" pycharm={"name": "#%%\n"}
coef[coef["Parámetro"].between(-1, 1)]

# %% colab={"base_uri": "https://localhost:8080/"} id="6gZhsvIYXJOf" outputId="8a9fe673-463a-46a5-e780-e476c287a42b" pycharm={"name": "#%%\n"}
mlflow.sklearn.log_model(best_model, "taller_1_model")

# %% pycharm={"name": "#%%\n"}
