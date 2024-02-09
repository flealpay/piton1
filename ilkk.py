import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = sns.load_dataset('titanic')
df.head()

df['sex'].value_counts()
#########################################
# Görev 2: Yukarıda tanımlanan Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
#########################################

df['sex'].value_counts().plot(kind='bar')
plt.show(block=True)

sns.countplot(x=df['sex'], data=df)
plt.show(block=True)

#########################################
# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
#########################################

df.nunique().sort_values(ascending=False)

#########################################
# Görev 4: pclass değişkeninin unique değerleri bulunuz.
#########################################

df['pclass'].unique()  # essiz degerleri
df['pclass'].nunique()  # essiz degerlerinin sayisi

#########################################
# Görev 5:  pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
#########################################

df[['pclass', 'parch']].nunique()

#########################################
# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz. Tekrar tipini kontrol ediniz.
#########################################

df['embarked'].dtype
df.embarked = df.embarked.astype('category')
df['embarked'].dtype

#########################################
# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
#########################################

df[df['embarked'] == 'C'].head()

#########################################
# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
#########################################

df[df['embarked'] != 'S'].head()

#########################################
# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
#########################################

df[(df['age'] < 30) & (df['sex'] == 'female')].head()

#########################################
# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
#########################################

df[(df['fare'] > 500) | (df['age'] > 70)]

#########################################
# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
#########################################

df.isnull().sum().sort_values(ascending=False)

#########################################
# Görev 12: who değişkenini dataframe'den düşürün.
#########################################

df = df.drop('who', axis=1).head()
df.head()

#########################################
# Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
#########################################

df['deck'].mode()
df['deck'].fillna(df['deck'].mode()[0]).head()

#########################################
# Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurun.
#########################################

x = df['age'].median()
df['age'].fillna(x)

#########################################
# Görev 15: survived değişkeninin Pclass ve Cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
#########################################

df.groupby(['pclass', 'sex']).agg({'survived': ['sum', 'count', 'mean']})

#########################################
# Görev 16:  30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazınız.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)
#########################################

def age_30(age):
    if age > 30:
        return 1
    else:
        return 0

df['age_flag'] = df['age'].apply(lambda x: age_30(x))
df.head()

df['new_age_flag'] = df['age'].apply(lambda x: 1 if x > 30 else 0)
df.head()


def age_30(age):

    df['age_flag_2'] = [1 if age < 30 else 0 for age in df['age']]

df['age'].apply(lambda x: age_30(x))
df.head()
#########################################
# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
#########################################

df = sns.load_dataset('tips')
df.head()

#########################################
# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill  değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

df.groupby('time').agg({'total_bill': ['sum', 'min', 'max', 'mean']})

#########################################
# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

df.groupby(['day', 'time']).agg({'total_bill': ['sum', 'min', 'max', 'mean']})

#########################################
# Görev 20:Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
#########################################

df[(df['time'] == 'Lunch') & (df['sex'] == 'Female')].groupby('day').agg({
    'total_bill': ['sum', 'min', 'max', 'mean'], 'tip': ['sum', 'min', 'max', 'mean']
})

#########################################
# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?
#########################################

df.loc[(df['size'] < 3) & (df['total_bill'] > 10), 'total_bill'].mean()
df['total_bill'][(df['size'] < 3) & (df['total_bill'] > 10)].mean()

#########################################
# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturun. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
#########################################

df['total_bill_tip_sum'] = df['total_bill'] + df['tip']
df.head()

#########################################
# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
#########################################

new_df = df.sort_values('total_bill_tip_sum', ascending=False).iloc[:30]
new_df