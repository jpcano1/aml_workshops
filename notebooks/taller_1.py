import setup_colab_general as setup_general

setup_general.setup_general()
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-deep")
# Librerías extras
import itertools
from typing import Optional

import pandas_profiling
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tabulate import tabulate

from utils import general as gen

# %% colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["8b89567e354a4e40a1172f82916a511c", "e8b207d9f4264815a5eb80159e03ddb1", "fc31b040db6c4d4aa48c642408bb1dd4", "31dee1517710491bb38383b2de73c64d", "4cbc8c79094942caa11b2a5c56395c7c", "4ff3b90cdb4e477995c1c542be1d247f", "906cd9ed90ad47858431e96c0d1a2795", "211dd72ef9444b07917d587a66792702", "be1538ff3fd64530bbefe18ecd152d3e", "a83080648e6942ab99ad26b7ab30d62e", "89e6287d42794cfe9e0e8833f3401dc0"]} executionInfo={"elapsed": 1241, "status": "ok", "timestamp": 1645572230131, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="CtzhV51Hwa5p" outputId="ca62a01d-d1a9-4d6f-935a-806dba6f63b6"
data_url = (
    "https://raw.githubusercontent.com/"
    "Camilojaravila/202210_MINE-4206_ANAL"
    "ISIS_CON_MACHINE_LEARNING/main/Taller%"
    "201/russian_prices.csv"
)
gen.download_content(data_url, filename="russian_prices.csv")

# %% [markdown] id="tvVbLTlBE6mh"
# A continuación, se leen los datos y se revisan las primeras líneas para verficar que la carga fue exitosa

# %% executionInfo={"elapsed": 160, "status": "ok", "timestamp": 1645574085863, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="kj0nEuMvxWad"
russian_prices_df = pd.read_csv("data/russian_prices.csv")

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"elapsed": 245, "status": "ok", "timestamp": 1645574086316, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="2wS1sFwMj6uO" outputId="29f52ad1-51fe-4ab6-8e9b-35481abe4192"
russian_prices_df.head()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1645574087350, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="wfg_2vpIj8-_" outputId="56f84ce2-f61a-479a-89f8-4d85b12eb890"
russian_prices_df.info()

# %% executionInfo={"elapsed": 147, "status": "ok", "timestamp": 1645574091462, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="Wl1BKpS4xAw2"
profiler = pandas_profiling.ProfileReport(russian_prices_df, dark_mode=True)

# %% [markdown] id="zfOPUCD0wNxc"
# - El perfilamiento se encuentra en los anexos.

# %% colab={"base_uri": "https://localhost:8080/", "height": 145, "referenced_widgets": ["1b2ac17108874ba4a03702d74c083dc1", "d57be4b11b4e4443ac2fe1436cf1d633", "c1a882ef442d4c199d9e88773fe54b26", "fc9a3ee5d69741128e51dfe33943eb5f", "be84857a1576469db69b3a111f2e3c81", "f6a81f831a3e439e81d1200b6e0a6c22", "6ad903acb3df43e39cc2f1e4d19a8da9", "2bf6ccfe45994627aa524c74340388eb", "cef2e0d6e9314b96b031cbad27c49443", "ce7a820eda4c4a268334d7f4bb255145", "30ce20111cd14d43b73d6c050e6e1ae5", "1dc78e23549e4f23a5c8b7270870e848", "8795a8b7db3845209ef5ac9d059b6b9d", "3347118a46cf49849f14a63968123bb5", "ae4085daf7974802b08a130c68fc5822", "37c66de59f37462bbe83534a584ee2a3", "8eab4d065a674a8782bab260ea09c4f2", "27067ce81281487ca1d82c1f5a0b2b9a", "133eaa5acb994a4abeb6a64dbae8b997", "57258020a1fb4cf1a68b2b728f42aee0", "6ceff0541d0244a694ed328d766fe1ad", "8034b9d4d15341b2925b24402be424c9", "9f3657a8b98e4728a1b3d98116550981", "d177c3f540a8461ebfa238c19cbba19d", "6364297eeda24bd29e13ae6234e5d9c2", "b416ee454d844e3f8176567b352807c8", "4e27d92afd8449648837dfbbff484417", "8cd0a2b0061d4e69aa96639c03df6360", "d0a8733532a845c68bde44a48b3b84b2", "19df0d3f78db4cd5a6560a4536e14232", "41e01814145d4c8c97e1e51b49d558d6", "1d8538337bc74194849c4ef1587c6255", "5dce4f679e0b44a7bddd69c737ce997f", "467c5676b03844a290f31c008afce209", "f2f5a385f0d6456eaf01bd6b38b83641", "b780269ebd68448cbf51d3969da1f122", "db8a2cb9aca442b88196deffae9f4d7e", "b81c840d52fd4b48b8bb77c33339380c", "a45bd7fed3ab40b2ae9da19bea0051be", "9e12d145b4194735b50cd0e175da6662", "f8796a8e4f5445609938d6ee9f7ab925", "337e212047e649b0ac6a918e0c6e8381", "94dc15d966db4637b335c61bd2f7c2ec", "492240f2ce6c46c09423ba877ed783cd"]} executionInfo={"elapsed": 49934, "status": "ok", "timestamp": 1645574142258, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="ztSi0cuPxffs" outputId="d0525cc1-b35f-4841-81a4-5d355736c828"
if not os.path.exists("profiling_reports"):
    os.makedirs("profiling_reports")
profiler.to_file("profiling_reports/russian_prices_profile.html")

# %% [markdown] id="oO11RRE2VWAN"
# - Las siguientes columnas se eliminaron bajo el supuesto de que no son necesarias para el objetivo de negocio. La primera columna es un identificador de propiedad, ergo, no es significativa. Las columnas `time` y `date` son columnas relacionadas a la publicación, más no a la propiedad perse, por lo tanto, no son significativas para nuestro modelo.

# %% executionInfo={"elapsed": 36, "status": "ok", "timestamp": 1645574142259, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="xUQji5iQi4oY"
columns_to_delete = [
    "Unnamed: 0",
    "time",
    "date",
]

# %% executionInfo={"elapsed": 34, "status": "ok", "timestamp": 1645574142260, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="HuVpJcPpjox-"
russian_prices_df.drop(columns=columns_to_delete, inplace=True)

# %% [markdown] id="k9tWqKeNsACK"
# - Todas las columnas con valores nulos fueron removidas

# %% executionInfo={"elapsed": 33, "status": "ok", "timestamp": 1645574142260, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="rdgm3ridkZhS"
russian_prices_df.dropna(inplace=True)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 32, "status": "ok", "timestamp": 1645574142260, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="2fvSWw_hj1wS" outputId="8053a42a-3ee2-4a43-a733-40e5bbaf4fbb"
russian_prices_df.info()

# %% executionInfo={"elapsed": 159, "status": "ok", "timestamp": 1645574146783, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="NXiS5wFokqHi"
russian_prices_df = russian_prices_df.apply(lambda x: x.astype("int32"))
russian_prices_df["object_type"] = russian_prices_df["object_type"].apply(
    lambda x: 2 if x == 11 else x
)
russian_prices_df["rooms"] = russian_prices_df["rooms"].apply(
    lambda x: -1 if x == -2 else x
)

# %% executionInfo={"elapsed": 192, "status": "ok", "timestamp": 1645574148264, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="dYivdCPnwfuo"
rows_to_drop = russian_prices_df.query(
    "kitchen_area + 5 >= area | area <= 10 | price <= 2000"
).index
russian_prices_df = russian_prices_df.drop(rows_to_drop).reset_index(drop=True)

# %% executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1645574148456, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="V7BDrFLJmcN9"
X, y = russian_prices_df.drop("price", axis=1), russian_prices_df["price"]

# %% executionInfo={"elapsed": 145, "status": "ok", "timestamp": 1645574149733, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="Gh_LIlQTmm0S"
full_X_train, X_test, full_y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)
X_train, X_val, y_train, y_val = train_test_split(
    full_X_train, full_y_train, test_size=0.2, random_state=1234
)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1645574149953, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="eEtwMvsAmogC" outputId="19421fab-345f-4dfc-c19b-f8ca0e9f6a41"
X_train.shape, y_train.shape

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 163, "status": "ok", "timestamp": 1645574152194, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="M3JwOPusnIpn" outputId="4814fcad-3e74-402f-84f7-f8754b00e18f"
X_val.shape, y_val.shape

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1645574152360, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="0QveAb5em00v" outputId="9db249db-3c83-4469-b77f-0e8b2e325c4c"
X_test.shape, y_test.shape


# %% [markdown] id="rP-CN-yeoJ_9"
# ## **Modelamiento**

# %% [markdown] id="16UKiXe_TM12"
# ### **Regresión Polinómial**
# #### **Entrenamiento**

# %% [markdown] id="zHyXQXfb2qAG"
# Se define la clase para realizar la transformación polinomial de nuetras variables

# %% executionInfo={"elapsed": 149, "status": "ok", "timestamp": 1645574153990, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="1EeXFbK_pP7a"
class ToPolynomial(BaseEstimator, TransformerMixin):
    def __init__(self, k: int = 2) -> None:
        self.k = k

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        columns = X.columns
        X_train_pol = pd.concat(
            [X ** (i + 1) for i in range(self.k)], axis=1
        )  # Polinomios sin interacciones
        X_train_pol.columns = np.reshape(
            [[i + " " + str(j + 1) for i in columns] for j in range(self.k)], -1
        )
        temp = pd.concat(
            [X[i[0]] * X[i[1]] for i in list(itertools.combinations(columns, 2))],
            axis=1,
        )  # Combinaciones sólo de grado 1
        temp.columns = [" ".join(i) for i in list(itertools.combinations(columns, 2))]
        X_train_pol = pd.concat([X_train_pol, temp], axis=1)
        return X_train_pol


# %% [markdown] id="Fj3hG4RM2wLO"
# Se crea un pipeline para encapsular los pasos de entrenamiento de nuestro modelo. Primero se realiza la transformación polinamial de las variables y estas se utilizan para entrenar el modelo de regresión lineal.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 280, "status": "ok", "timestamp": 1645574156411, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="2yMSJKDloOZi" outputId="8d1006f7-2e7a-46a7-82a4-7a5e1c059fac"
estimators = [("polinomial", ToPolynomial()), ("regresion", LinearRegression())]

pipe_pol = Pipeline(estimators)

pipe_pol.fit(X_train, y_train)

# %% [markdown] id="0G2Ruy0T4DUD"
#  Parámetros entrenados por la Regresión Polinomial

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1645574157051, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="eCPY09nsTcWK" outputId="d999de95-85a3-49f6-ca05-f1c73f1fdcdd"
reg_lineal = pipe_pol["regresion"]

print("Intercept: ", reg_lineal.intercept_)
print("Coefficients: ", reg_lineal.coef_)

# %% [markdown] id="epQPSIhSTWXo"
# #### **Validación**

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 270, "status": "ok", "timestamp": 1645574159588, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="-epy5ZCTT2hd" outputId="96f6cb63-c24b-4d9a-80fa-14631dd49364"
y_pred = pipe_pol.predict(X_val)
y_pred

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1645574159756, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="UzYpZljfT9qv" outputId="6418c3ba-368d-4a37-ddfa-e5388e7e7eea"
r2_poly = r2_score(y_val, y_pred)
mse_poly = mean_squared_error(y_val, y_pred)
mae_poly = mean_absolute_error(y_val, y_pred)

print("------------ Polynomial Regression ------------")
print(f"R2-score: {r2_poly:.7f}")
print(f"Residual sum of squares (MSE): {mse_poly:.5f}")
print(f"Mean absolute error: {mae_poly:.5f}")


# %% [markdown] id="hWv4uZTyeZ1l"
# #### **Comportamiento de los datos reales vs los datos predecidos**

# %% executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1645574162010, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="lt_fp3tjdVJy"
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


# %% colab={"base_uri": "https://localhost:8080/", "height": 729} executionInfo={"elapsed": 802, "status": "ok", "timestamp": 1645574164620, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="HA1FneL3hcOX" outputId="b042285d-8e7f-4a9d-91c3-5d2da027213e"
draw_chart(y_val, y_pred, "Predicción con Regresión Polinomial", "Regresión Polinomial")

# %% [markdown] id="ud_HEwjZoOrq"
# ### **Regresión Ridge**
# #### **Entrenamiento (Sin estandarización)**
#
#

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 169, "status": "ok", "timestamp": 1645574167726, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="rnnb_rWBoRHa" outputId="907ff1e6-939a-4077-d745-011977d8997f"
ridge_reg = Ridge()
ridge_reg.fit(X_train, y_train)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1645574167890, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="xe1WtO-IiepT" outputId="fce0d49b-d8d0-4a21-96c0-6184057903b3"
ridge_coef = dict(zip(X_train.columns, ridge_reg.coef_))
ridge_coef

# %% [markdown] id="Zdx6xzlXdN0b"
# #### **Validación**

# %% executionInfo={"elapsed": 163, "status": "ok", "timestamp": 1645574172582, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="26Wmz50fdRx5"
y_pred_2 = ridge_reg.predict(X_val)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 209, "status": "ok", "timestamp": 1645574172977, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="EvQB0EHsi_Fm" outputId="5a95ae13-ff7c-473f-94a8-a9fad90676ec"
r2_ridge = r2_score(y_val, y_pred_2)
mse_ridge = mean_squared_error(y_val, y_pred_2)
mae_ridge = mean_absolute_error(y_val, y_pred_2)

print("------------ Ridge ------------")
print(f"R2-score: {r2_ridge:.7f}")
print(f"Residual sum of squares (MSE): {mse_ridge:.5f}")
print(f"Mean absolute error: {mae_ridge:.5f}")

# %% [markdown] id="UkIx8aTSe8JA"
# #### **Comportamiento de los datos reales vs los datos predecidos**

# %% colab={"base_uri": "https://localhost:8080/", "height": 729} executionInfo={"elapsed": 959, "status": "ok", "timestamp": 1645574177978, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="bkaeBDIuevsV" outputId="cb4f1aa1-2910-4c7c-e714-94fe6b5d31cb"
# %matplotlib inline
draw_chart(y_val, y_pred_2, "Predicción con regresion Ridge", "Regresion Ridge")

# %% [markdown] id="9Pd-sS9eYGzD"
# #### **Entrenamiento (Con estandarización)**

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 157, "status": "ok", "timestamp": 1645574187308, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="wcsl9CmhYJ_7" outputId="80ee7a4b-9df9-456c-83ac-b74e133c36e7"
pipeline_ridge = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("regressor", Ridge()),
    ],
)

pipeline_ridge.fit(X_train, y_train)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1645574197207, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="lLkPRJlSZZvS" outputId="939dbe26-e732-4f6c-f3b8-7a3c0e0e0156"
ridge_coef = dict(zip(X_train.columns, pipeline_ridge.steps[1][1].coef_))
ridge_coef

# %% [markdown] id="dflufl4waDcF"
# #### **Validación**

# %% executionInfo={"elapsed": 183, "status": "ok", "timestamp": 1645574200454, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="6SYHTEbBaHiw"
y_pred_2b = pipeline_ridge.predict(X_val)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1645574200606, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="EelZTCKzaMSL" outputId="ce5e0dc3-2097-405c-e5c0-1387e608e5a7"
r2_ridge_s = r2_score(y_val, y_pred_2b)
mse_ridge_s = mean_squared_error(y_val, y_pred_2b)
mae_ridge_s = mean_absolute_error(y_val, y_pred_2b)

print("------------ Ridge (Con estandarización) ------------")
print(f"R2-score: {r2_ridge_s:.7f}")
print(f"Residual sum of squares (MSE): {mse_ridge_s:.5f}")
print(f"Mean absolute error: {mae_ridge_s:.5f}")

# %% [markdown] id="rbeZ9Bc2gxRu"
# #### **Comportamiento de los datos reales vs los datos predecidos**

# %% colab={"base_uri": "https://localhost:8080/", "height": 729} executionInfo={"elapsed": 907, "status": "ok", "timestamp": 1645574206676, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="JRh9LRiNigzY" outputId="e9d8057f-92a8-4871-c317-f6395b2669ad"
# %matplotlib inline
draw_chart(
    y_val,
    y_pred_2b,
    "Predicción con regresion Ridge (Con estandarización)",
    "Regresion Ridge",
)

# %% [markdown] id="mQ_FoF7RoRX9"
# ### **Regresión Lasso**
# #### **Entrenamiento (Sin estandarización)**

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 194, "status": "ok", "timestamp": 1645574212938, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="TLdd00Wdm69O" outputId="ade54953-1ed2-43d6-e054-af77c9a60dc7"
lasso_reg = Lasso()
lasso_reg.fit(X_train, y_train)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1645574213168, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="atcLgUTkoaKF" outputId="96ad5fb1-57f6-426f-8bd6-262ed3f92248"
lasso_coef = dict(zip(X_train.columns, lasso_reg.coef_))
lasso_coef

# %% [markdown] id="NE0rheOho8Oj"
# #### **Validación**

# %% executionInfo={"elapsed": 152, "status": "ok", "timestamp": 1645574215952, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="yRCL86NIokX6"
y_pred_3 = lasso_reg.predict(X_val)

r2_lasso = r2_score(y_val, y_pred_3)
mse_lasso = mean_squared_error(y_val, y_pred_3)
mae_lasso = mean_absolute_error(y_val, y_pred_3)

print("------------ Lasso ------------")
print(f"R2-score: {r2_lasso:.4f}")
print(f"Residual sum of squares (MSE): {mse_lasso:.5f}")
print(f"Mean absolute error: {mae_lasso:.5f}")


draw_chart(y_val, y_pred_3, "Predicción con regresion Lasso", "Regresion Lasso")

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("regressor", Lasso()),
    ],
)

pipeline.fit(X_train, y_train)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1645574236215, "user": {"displayName": "Juan Pablo Cano", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ggpvl16I60w7jnrbdVFshSpDWtSuXlYdGvZAOpQXQ=s64", "userId": "14080729078587151746"}, "user_tz": 300} id="QuqmS5LTdjr9" outputId="4a28097b-ad82-4751-f11c-ded4ea6c0760"
lasso_coef = dict(zip(X_train.columns, pipeline.steps[1][1].coef_))
lasso_coef


y_pred_3b = pipeline.predict(X_val)

r2_lasso_s = r2_score(y_val, y_pred_3b)
mse_lasso_s = mean_squared_error(y_val, y_pred_3b)
mae_lasso_s = mean_absolute_error(y_val, y_pred_3b)

print("------------ Lasso (Con estandarización) ------------")
print(f"R2-score: {r2_lasso_s:.7f}")
print(f"Residual sum of squares (MSE): {mse_lasso_s:.5f}")
print(f"Mean absolute error: {mae_lasso_s:.5f}")


draw_chart(
    y_val,
    y_pred_3b,
    "Predicción con regresion Lasso (Con estandarización)",
    "Regresion Lasso",
)


info = {
    "Model": ["Poly Regression", "Ridge", "Ridge (con S)", "Lasso", "Lasso (con S)"],
    "R2": [r2_poly, r2_ridge, r2_ridge_s, r2_lasso, r2_lasso_s],
    "MSE": [mse_poly, mse_ridge, mse_ridge_s, mse_ridge, mse_lasso_s],
    "MAE": [mae_poly, mae_ridge, mae_ridge_s, mae_lasso, mae_lasso_s],
}

print(tabulate(info, headers="keys", tablefmt="fancy_grid"))
