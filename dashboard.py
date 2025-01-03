import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from statsmodels.tsa.arima.model import ARIMA

# Upload File Excel
st.sidebar.header("Upload File Excel")
uploaded_file = st.sidebar.file_uploader("Pilih file Excel", type=['xlsx'])

# Baca Data dari File yang Diunggah
if uploaded_file:
    data = pd.read_excel(uploaded_file)
else:
    data = pd.read_excel('data_penjualan_3bulan.xlsx')

# Konversi tanggal ke datetime
data['tanggal'] = pd.to_datetime(data['tanggal'])
data.set_index('tanggal', inplace=True)

# Sidebar untuk Filter Data
st.sidebar.header("Filter Data")
start_date = st.sidebar.date_input("Start Date", value=data.index.min())
end_date = st.sidebar.date_input("End Date", value=data.index.max())

filtered_data = data[(data.index >= pd.Timestamp(start_date)) & (data.index <= pd.Timestamp(end_date))]

# Filter Berdasarkan Produk
product_filter = st.sidebar.multiselect("Pilih Produk", data['nama_produk'].unique(), default=data['nama_produk'].unique())
filtered_data = filtered_data[filtered_data['nama_produk'].isin(product_filter)]

# Tampilkan Data di Dashboard
st.title("Dashboard Penjualan dengan Prediksi")
st.write("Data penjualan yang ditampilkan berdasarkan filter yang dipilih.")

st.dataframe(filtered_data)

# Visualisasi Data Penjualan
st.subheader("Jumlah Penjualan per Tanggal")
fig, ax = plt.subplots()
for product in product_filter:
    product_data = filtered_data[filtered_data['nama_produk'] == product]
    ax.plot(product_data.index, product_data['jumlah'], label=f"{product} (Aktual)")

ax.set_xlabel("Tanggal")
ax.set_ylabel("Jumlah Penjualan")
ax.legend()
st.pyplot(fig)

# Prediksi Moving Average & ARIMA
st.subheader("Prediksi Penjualan (Moving Average dan ARIMA)")
window_size = st.slider("Window Size (Moving Average)", min_value=3, max_value=14, value=7)

for product in product_filter:
    product_data = filtered_data[filtered_data['nama_produk'] == product].resample('D').sum()
    product_data['ma'] = product_data['jumlah'].rolling(window=window_size).mean()
    
    # Prediksi ARIMA jika data mencukupi
    if len(product_data.dropna()) > 10:
        model = ARIMA(product_data['jumlah'].dropna(), order=(5, 1, 0))
        arima_result = model.fit()
        future_forecast = arima_result.forecast(steps=30)
        ax.plot(future_forecast.index, future_forecast, '--', label=f"{product} (ARIMA Prediksi)")
    
    ax.plot(product_data.index, product_data['ma'], label=f"{product} (Moving Average)")

ax.legend()
st.pyplot(fig)

# ===========================
# DOWNLOAD LAPORAN (PDF/Excel)
# ===========================

def download_excel():
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        filtered_data.to_excel(writer, sheet_name='Data Penjualan')
    processed_data = output.getvalue()
    return processed_data

st.sidebar.subheader("Download Laporan")
download_format = st.sidebar.radio("Pilih format laporan:", ('Excel', 'PDF'))

if st.sidebar.button("Download"):
    if download_format == 'Excel':
        st.sidebar.download_button(
            label="Download Excel",
            data=download_excel(),
            file_name="laporan_penjualan.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        buffer = BytesIO()
        plt.savefig(buffer, format='pdf')
        buffer.seek(0)
        st.sidebar.download_button(
            label="Download PDF",
            data=buffer,
            file_name="laporan_penjualan.pdf",
            mime="application/pdf"
        )
