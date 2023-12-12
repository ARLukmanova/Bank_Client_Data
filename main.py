import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    layout='wide',
    page_title="Разведочный анализ данных",
)

df = pd.read_csv('clients_data.csv')

st.header('Разведочный анализ данных')
st.subheader('Данные о клиентах и их описание ')

st.dataframe(df)

st.markdown(
    """
    Набор данных включает в себя следующие характеристики клиента:

- AGE - возраст клиента;
- GENDER - пол клиента (1 — мужчина, 0 — женщина);
- EDUCATION - уровень образования клиента;
- MARITAL_STATUS - семейное положение клиента;
- CHILD_TOTAL - количество детей у клиента;
- DEPENDANTS - количество человек на иждивении у клиента;
- SOCSTATUS_WORK_FL - трудоустроен ли клиент (1 — работает, 0 — не работает);
- SOCSTATUS_PENS_FL - является ли клиент пенсионером (1 — пенсионер, 0 — не пенсионер);
- REG_ADDRESS_PROVINCE - регион регистрации клиента;
- FACT_ADDRESS_PROVINCE - регион фактического пребывания клиента;
- POSTAL_ADDRESS_PROVINCE - регион в составе почтового адреса клиента;
- FL_PRESENCE_FL - является ли клиент владельцем квартиры (1 — есть, 0 — нет);
- OWN_AUTO - количество автомобилей в собственности у клиента;
- TARGET - (целевая переменная) отклик клиента на маркетинговую кампанию (1 — отклик был, 0 — отклика не было);
- FAMILY_INCOME - семейный доход клиента (один из нескольких диапазонов);
- PERSONAL_INCOME - личный доход клиента (в рублях);
- CREDIT - размер последнего кредита клиента (в рублях);
- TERM - срок последнего кредита клиента;
- FST_PAYMENT - размер первоначального взноса по последнему кредиту клиента (в рублях);
- LOAN_NUM_TOTAL - количество кредитов в кредитной истории клиента;
- LOAN_NUM_CLOSED  - количество закрытых кредитор в кредитной истории клиента."""
)

st.subheader('Характеристики признаков')
st.dataframe(df.describe(include='all').T, width=1200, height=780)
st.markdown("""Всего записей 15176, отсутствуют пропуски. 
Приведена информация для числовых признаков:  среднее значение, стандартное отклонение, минимум, максимум, медиана и 25/75 перцентили, 
для категориальных: количество уникальных категорий и наиболее популярная.""")


st.subheader('Целевая переменная')
figure = plt.figure(figsize=(3, 2))
sns.histplot(df, x='TARGET', hue='TARGET')
st.pyplot(figure, use_container_width=False)
plt.close()
st.markdown(
    f'Видим, что откликнувшихся на предложение банка в выборке сильно меньше: **{df["TARGET"].sum() / df["TARGET"].count():.2%}**')

st.subheader('Возраст клиента')

figure = plt.figure(figsize=(3, 2))
mean_age = df['AGE'].mean()
plt.axvline(mean_age, color='red', linewidth=2, label=f'Mean Age: {mean_age:.2f}')
sns.histplot(df, x='AGE', bins=15, kde=True)
plt.legend(loc='upper right', fontsize='xx-small')
st.pyplot(figure, use_container_width=False)
plt.close()

min_age = df['AGE'].min()
st.markdown(
    f"""
    Мы видим, что возраст клиентов находится в диапазоне между {df['AGE'].min()} и {df['AGE'].max()}, среднее значение возраста равно {mean_age:.2f}.
    """
)

st.subheader('Личный доход и размер последнего кредита')

figure = plt.figure(figsize=(14, 4))
axes = figure.subplots(1, 2)
g = sns.histplot(df, x="PERSONAL_INCOME", kde=True, bins=15, ax=axes[0])
g = sns.histplot(df, x="CREDIT", kde=True, bins=15, ax=axes[1])
figure.tight_layout()
st.pyplot(figure, use_container_width=False)  # выводим на форму
plt.close()  # закрываем, т.к. потом всякий сообщения в консоли лезут

st.markdown(
    f'Распределения дохода и размера последнего кредита сильно смещены влево (имеют логнормальное распределение).')

st.subheader('Количество взятых и закрытых кредитов')

figure = plt.figure(figsize=(14, 4))
axes = figure.subplots(1, 2)
g = sns.histplot(df, x="LOAN_NUM_TOTAL", discrete=True, ax=axes[0])
g = sns.histplot(df, x="LOAN_NUM_CLOSED", discrete=True, ax=axes[1])
figure.tight_layout()
st.pyplot(figure, use_container_width=False)  # выводим на форму
plt.close()  # закрываем, т.к. потом всякий сообщения в консоли лезут
st.markdown(
    f'В основном у клиентов банка количество взятых и закрытых кредитов не превышает 2.')

st.subheader('Распределение количества детей и иждивенцев')
figure = plt.figure(figsize=(14, 4))
axes = figure.subplots(1, 2)
g = sns.histplot(df, x="CHILD_TOTAL", discrete=True, shrink=0.8, ax=axes[0])
g = sns.histplot(df, x="DEPENDANTS",  discrete=True, shrink=0.8, ax=axes[1])
figure.tight_layout()
st.pyplot(figure, use_container_width=False)  # выводим на форму
plt.close()  # закрываем, т.к. потом всякий сообщения в консоли лезут
st.markdown(
    f'Большинство клиентов банка имеют не более 3 детей, а на иждивении не более 2.')

st.subheader('Корреляция числовых признаков')

figure = plt.figure(figsize=(12, 5))
sns.heatmap(df[[
    'AGE',
    'CHILD_TOTAL',
    'DEPENDANTS',
    'OWN_AUTO',
    'PERSONAL_INCOME',
    'CREDIT',
    'TERM',
    'FST_PAYMENT',
    'LOAN_NUM_TOTAL',
    'LOAN_NUM_CLOSED']].corr(), annot=True)
figure.tight_layout()
st.pyplot(figure, use_container_width=False)  # выводим на форму
plt.close()  # закрываем, т.к. потом всякий сообщения в консоли лезут
st.markdown(
    """
    - Имеется положительная корреляция между возрастом клиента и общим количеством детей и обратная с количеством детей на иждивении. Этот факт вполне объясним.
    - Имеется существенная корреляция между доходом клиента и наличием авто, размером кредита и величиной первого платежа по нему. Чем большим доходом обладает клиент, 
    тем проще позволить себе машину и взять кредит.
    
    - Имеется значительная корреляция между размером кредита и величиной первого платежа, сроком последнего кредита. Чем более существенным является кредит, 
    тем более предпочтительным является длительный срок и большой размер первоначального взноса.""")

st.subheader('Зависимость целевой переменной от непрерывных числовых признаков')

figure = plt.figure(figsize=(3, 2))
sns.histplot(df, x='AGE', bins=15, hue='TARGET', multiple='fill')
st.pyplot(figure, use_container_width=False)
plt.close()
st.markdown(
    "С возрастом доля людей, откликнувшихся на предложение банка, уменьшается."    )

st.subheader('Зависимость целевой переменной от личного дохода и размера последнего кредита')

figure = plt.figure(figsize=(14, 4))
axes = figure.subplots(1, 2)
g = sns.histplot(df, x="PERSONAL_INCOME", hue='TARGET', multiple='fill', bins=15, ax=axes[0])
g = sns.histplot(df, x="CREDIT", hue='TARGET', multiple='fill', bins=20, ax=axes[1])
figure.tight_layout()
st.pyplot(figure, use_container_width=False)  # выводим на форму
plt.close()  # з
st.markdown(
    "С ростом дохода до 40 000 доля людей, откликнувшихся на предложение банка, увеличивается."    )
st.markdown(
    "С увеличением размера последнего кредита, доля откликнувшихся на предложение банка, увеличивается."    )

st.subheader('Зависимость целевой переменной от количество взятых и закрытых кредитов')

figure = plt.figure(figsize=(14, 4))
axes = figure.subplots(1, 2)
g = sns.histplot(df, x="LOAN_NUM_TOTAL", hue='TARGET', multiple='fill', discrete=True, shrink=0.8, ax=axes[0])
g = sns.histplot(df, x="LOAN_NUM_CLOSED", hue='TARGET', multiple='fill', discrete=True, shrink=0.8, ax=axes[1])
figure.tight_layout()
st.pyplot(figure, use_container_width=False)  # выводим на форму
plt.close()  # закрываем, т.к. потом всякий сообщения в консоли лезут

st.subheader('Зависимость целевой переменной от количества детей и иждивенцев')
figure = plt.figure(figsize=(14, 4))
axes = figure.subplots(1, 2)
g = sns.histplot(df, x="CHILD_TOTAL", hue='TARGET', multiple='fill', discrete=True, shrink=0.8, ax=axes[0])
g = sns.histplot(df, x="DEPENDANTS", hue='TARGET', multiple='fill', discrete=True, shrink=0.8, ax=axes[1])
figure.tight_layout()
st.pyplot(figure, use_container_width=False)  # выводим на форму
plt.close()  # закрываем, т.к. потом всякий сообщения в консоли лезут

st.subheader('Зависимость целевой переменной от значения бинарных признаков')

figure = plt.figure(figsize=(10, 4))
axes = figure.subplots(2, 2)
g = sns.histplot(df, x="GENDER", hue='TARGET', multiple='fill', ax=axes[0, 0])
g.set_xticks(range(2))
g.set_xticklabels(['женщины', 'мужчины'])

g = sns.histplot(df, x="SOCSTATUS_WORK_FL", hue='TARGET', multiple='fill', ax=axes[0, 1])
g.set_xticks(range(2))
g.set_xticklabels(['не работающий', 'работающий'])

g = sns.histplot(df, x="SOCSTATUS_PENS_FL", hue='TARGET', multiple='fill', ax=axes[1, 0])
g.set_xticks(range(2))
g.set_xticklabels(['не пенсионер', 'пенсионер'])

g = sns.histplot(df, x="FL_PRESENCE_FL", hue='TARGET', multiple='fill', ax=axes[1, 1])
g.set_xticks(range(2))
g.set_xticklabels(['нет квартиры', 'есть квартира'])

figure.tight_layout()
st.pyplot(figure, use_container_width=False)  # выводим на форму
plt.close()  # закрываем, т.к. потом всякий сообщения в консоли лезут

st.markdown(
    """
    Получаем следующие выводы:
    + Среди мужчин доля откликнувшихся на предложение банка меньше.
    + Среди работающих доля откликнувшихся на предложение банка больше.
    + Среди не пенсионеров доля откликнувшихся на предложение банка больше.
    + Среди обладателей квартиры доля откликнувшихся на предложение банка чуть меньше.

    """
)

st.subheader('Зависимость целевой переменной других категориальных признаков')

figure = plt.figure(figsize=(14, 2))  # создаем фигуру матплота, и.к. именно она рисуется
sns.histplot(df, y="MARITAL_STATUS", hue='TARGET', multiple='fill')
st.pyplot(figure, use_container_width=False)  # выводим на форму
plt.xticks(fontsize=2)
plt.yticks(fontsize=2)
plt.close()  # закрываем, т.к. потом всякий сообщения в консоли лезут

st.markdown("Видим, что среди тех, кто состоит в гражданском браке, доля откликнувшихся на предложение банка больше.")

figure = plt.figure(figsize=(14, 2))  # создаем фигуру матплота, и.к. именно она рисуется
sns.histplot(df, y="FAMILY_INCOME", hue='TARGET', multiple='fill')
st.pyplot(figure, use_container_width=False)  # выводим на форму
plt.xticks(fontsize=2)
plt.yticks(fontsize=2)
plt.close()  # закрываем, т.к. потом всякий сообщения в консоли лезут

st.markdown("Видим, что среди тех, кто получает свыше 50 000 руб, доля откликнувшихся на предложение банка больше.")

figure = plt.figure(figsize=(12, 2))  # создаем фигуру матплота, и.к. именно она рисуется
sns.histplot(df, y="EDUCATION", hue='TARGET', multiple='fill')
st.pyplot(figure, use_container_width=False)  # выводим на форму
plt.xticks(fontsize=2)
plt.yticks(fontsize=2)
plt.close()  # закрываем, т.к. потом всякий сообщения в консоли лезут

st.markdown(
    "Наибольшая доля людей, откликнувшихся на предложение банка, среди имеющих неоконченное высшее или два и более "
    "высших образования.")

figure = plt.figure(figsize=(10, 2))


def gr(x):
    return pd.Series({
        'COUNT': x['TARGET'].count(),
        'ACCEPTED': x['TARGET'].sum(),
        'LEAD_SHARE': x['TARGET'].sum() / x['TARGET'].count()
    })


agg_data = (df.groupby('FACT_ADDRESS_PROVINCE', as_index=False)
            .apply(lambda x: pd.Series({
    'SHARE': x['TARGET'].sum() / x['TARGET'].count()
})
                   )
            .sort_values('SHARE', ascending=False).head(10))

sns.barplot(agg_data, x='SHARE', y='FACT_ADDRESS_PROVINCE', color='orange')
st.pyplot(figure, use_container_width=False)  # выводим на форму
plt.xticks(fontsize=2)
plt.yticks(fontsize=2)
plt.close()  # закрываем, т.к. потом всякий сообщения в консоли лезут


st.markdown(
    "Наиболее существенными признаками, влияющими на целевую переменную являются возраст, доход и социальный статус клиента."    )
