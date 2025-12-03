import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AutoPred")  # над названием я долго не думал, да


@st.cache_resource
def load_artifacts(path="model_artifacts.pkl"):
    with open(path, "rb") as f:
        artifacts = pickle.load(f)
    return artifacts


@st.cache_data
def load_train_data(path="cars_train.csv"):
    return pd.read_csv(path)


def prep_data(df_raw, artifacts):
    df = prep_eda(df_raw)
    cat_cols = artifacts["cat_cols"]
    ohe = artifacts["ohe"]

    data_ohe = ohe.transform(df[cat_cols])

    ohe_cols = ohe.get_feature_names_out(cat_cols)

    df_ohe = pd.DataFrame(data_ohe, columns=ohe_cols, index=df.index)

    df_final = pd.concat([df.drop(columns=cat_cols), df_ohe], axis=1)

    return df_final


def prep_eda(df_raw):
    df = df_raw.copy()
    if "selling_price" in df:
        df.drop(columns="selling_price", inplace=True)
    cols_to_clean = ["mileage", "engine", "max_power"]

    for col in cols_to_clean:
        df[col] = df[col].astype(str).str.extract(r"(\d+\.?\d*)", expand=False).astype(float)

    if "torque" in df:
        df.drop(columns=["torque"], inplace=True)

    num_cols = df.select_dtypes(include="number").columns.tolist()
    medians = df[num_cols].median()
    df[num_cols] = df[num_cols].fillna(medians)

    df["engine"] = df["engine"].astype(int)
    df["seats"] = df["seats"].astype(int)
    df["brand"] = df["name"].str.split().str[0]

    if "name" in df:
        df.drop(columns="name", inplace=True)

    return df


artifacts = load_artifacts()

model = artifacts["model"]

st.title("Прогнозирования цены автомобиля")

page = st.sidebar.radio("Раздел", ["EDA", "Прогноз по CSV", "Ручной ввод", "Веса модели"])

df_train = load_train_data()
df_eda = pd.concat([prep_eda(df_train), df_train["selling_price"]], axis=1)
if page == "EDA":
    st.subheader("Общий вид данных")
    st.dataframe(df_train.head())
    st.subheader("Вид данных после обработки")
    st.dataframe(df_eda.head())
    st.subheader("Распределение целевой переменной (selling_price)")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df_eda["selling_price"], bins=50, ax=ax)
    ax.set_xlabel("selling_price")
    st.pyplot(fig)

    st.subheader("Распределение числового признака")
    num_col_sel = st.selectbox("Выберите числовой признак", artifacts["num_cols"])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df_eda[num_col_sel], bins=40, ax=ax)
    ax.set_xlabel(num_col_sel)
    st.pyplot(fig)

    st.subheader("Связь признака с ценой (scatter)")
    num_x = st.selectbox("Числовой признак по оси X", artifacts["num_cols"], key="scatter_num")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.regplot(data=df_eda, x=num_x, y="selling_price", ax=ax, line_kws=dict(color="r"), scatter_kws=dict(alpha=0.25))

    ax.set_ylabel("selling_price")
    st.pyplot(fig)

    st.subheader("Зависимость цены от категориального признака (boxplot)")
    cat_col_sel = st.selectbox("Категориальный признак", artifacts["cat_cols"])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df_eda, x=cat_col_sel, y="selling_price", ax=ax)
    ax.set_ylabel("selling_price")
    plt.xticks(rotation=90)
    st.pyplot(fig)

elif page == "Прогноз по CSV":
    st.subheader("Прогноз для набора объектов из CSV")
    st.write(
        "Ожидается CSV с теми же колонками, что и в обучающем датасете (минимум числовые и категориальные признаки, включая `name`)."
    )
    file = st.file_uploader("Загрузите CSV-файл", type="csv")
    if file is not None:
        df_new = pd.read_csv(file)
        st.write("Первые строки загруженного датасета:")
        st.dataframe(df_new.head())
        try:
            new_prep = prep_data(df_new, artifacts)
            preds = model.predict(new_prep)
            df_result = df_new.copy()
            df_result["pred_selling_price"] = preds
            st.subheader("Результаты прогноза")
            st.dataframe(df_result)
            csv_out = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Скачать CSV с предсказаниями", data=csv_out, file_name="predictions.csv", mime="text/csv"
            )
        except Exception as e:
            st.error(f"Ошибка при подготовке признаков: {e}")

elif page == "Веса модели":
    st.subheader("Визуализация весов модели")
    try:
        feature_names = model.feature_names_in_
        coefs = model.coef_
        coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
        coef_df["abs_coef"] = coef_df["coef"].abs()
        coef_df = coef_df.sort_values("abs_coef", ascending=False)

        top_n = st.slider("Сколько самых важных признаков показать", 5, max(5, len(coef_df)), 20)  # на всякий случай чисто
        top_df = coef_df.head(top_n).sort_values("coef")

        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
        ax.barh(top_df["feature"], top_df["coef"])
        ax.set_xlabel("Коэффициент")
        ax.set_ylabel("Признак")
        ax.set_title("Веса модели (Ridge)")
        st.pyplot(fig)

        st.subheader("Таблица коэффициентов")
        st.dataframe(coef_df[["feature", "coef"]].head(top_n))
    except Exception as e:
        st.error(f"Не удалось построить веса: {e}")
elif page == "Ручной ввод":
    st.subheader("Прогноз по одному автомобилю (ручной ввод)")

    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input("Год выпуска", min_value=1980, max_value=2025, value=2015, step=1)
        km_driven = st.number_input("Пробег, км", min_value=0, max_value=500000, value=60000, step=1000)
        mileage = st.number_input("Расход, kmpl", min_value=0.0, max_value=50.0, value=18.0, step=0.1)
        engine = st.number_input("Объем двигателя, cc", min_value=600, max_value=5000, value=1200, step=50)
        max_power = st.number_input("Мощность, bhp", min_value=0.0, max_value=400.0, value=80.0, step=1.0)

    with col2:
        fuel = st.selectbox("Тип топлива", sorted(df_eda["fuel"].unique()))
        seller_type = st.selectbox("Тип продавца", sorted(df_eda["seller_type"].unique()))
        transmission = st.selectbox("Коробка передач", sorted(df_eda["transmission"].unique()))
        owner = st.selectbox("Тип владельца", sorted(df_eda["owner"].unique()))
        seats = st.selectbox("Количество мест", sorted(df_eda["seats"].unique()))
        name = st.text_input("Название")

    if st.button("Предсказать цену"):
        row = {
            "year": year,
            "km_driven": km_driven,
            "mileage": mileage,
            "engine": engine,
            "max_power": max_power,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner,
            "seats": int(seats),
            "name": name,
        }

        df_one = pd.DataFrame([row])

        try:
            final = prep_data(df_one, artifacts)
            pred = model.predict(final)[0]
            st.success("Прогнозируемая цена: " + f"{pred:,.0f}".replace(",", "'"))
        except Exception as e:
            st.error(f"Ошибка при подготовке признаков или предсказании: {e}")
