import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load dataset
file_path = "BlaBla.xlsx"
df = pd.read_excel(file_path)
df = df.replace({"Y": 1, "N": 0})

st.title("Decision Tree Klasifikasi Pasien")

# 2. Menampilkan semua data
st.subheader("2. Menampilkan Semua Data")
st.dataframe(df.head())

# 3. Grouping
st.subheader("3. Grouping Data")
st.write(df.groupby("N").size())
st.write("Jumlah variabel fitur =", len(df.columns) - 2)
st.write("Nama variabel fitur =", list(df.drop(columns=["N", "A"]).columns))
st.write("Target kelas = N (0=Negative, 1=Positive)")

# 4. Training & Testing
X = df.drop(columns=["N", "A"])
y = df["N"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

st.subheader("4. Split Data Training & Testing")
st.write("Jumlah Data Training =", len(X_train))
st.write("Jumlah Data Testing =", len(X_test))

# 5. Decision Tree
clf = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
clf.fit(X_train, y_train)
st.subheader("5. Decision Tree")
st.success("Model berhasil dilatih.")

# 6. Prediksi & Instance Output
y_pred = clf.predict(X_test)
st.subheader("6. Instance Prediksi Decision Tree")
st.write("Beberapa hasil prediksi vs data asli:")
for i in range(5):
    st.write("Prediksi =", y_pred[i], " | Asli =", list(y_test)[i])

# 7. Akurasi
st.subheader("7. Akurasi Model")
st.write("Akurasi :", accuracy_score(y_test, y_pred))

# 8. Classification Report
st.subheader("8. Classification Report")
st.text(classification_report(y_test, y_pred))

# 9. Visualisasi Decision Tree
st.subheader("9. Visualisasi Pohon Keputusan")
fig, ax = plt.subplots(figsize=(16, 8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=["Negative", "Positive"])
st.pyplot(fig)

# 🔹 10. Form Input Pasien
st.subheader("10. Prediksi Pasien Baru")

umur = st.number_input("Umur Pasien", min_value=1, max_value=100, step=1)
gender = st.radio("Jenis Kelamin", ["Perempuan", "Laki-laki"])
gender = 0 if gender == "Perempuan" else 1

gejala = []
for kolom in X.columns[2:]:  # ambil kolom C–M
    val = st.radio(f"Apakah pasien mengalami {kolom}?", ["Tidak", "Ya"], index=0)
    gejala.append(1 if val == "Ya" else 0)

# Kategori umur (A_k)
if umur < 21:
    A_k = 1
elif umur <= 30:
    A_k = 2
elif umur <= 40:
    A_k = 3
elif umur <= 50:
    A_k = 4
else:
    A_k = 5

if st.button("Prediksi Pasien"):
    data_input = [A_k, gender] + gejala
    hasil = clf.predict([data_input])[0]
    st.success("Hasil Prediksi: **Positive**" if hasil == 1 else "Hasil Prediksi: **Negative**")
    st.write("Kode umur =", A_k)
    st.write("Data input =", data_input)
