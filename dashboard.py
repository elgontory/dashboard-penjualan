import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load data from Excel
file_path = 'data_penjualan_3bulan.xlsx'
data = pd.read_excel(file_path)

# Convert tanggal to datetime
data['tanggal'] = pd.to_datetime(data['tanggal'])
data.set_index('tanggal', inplace=True)

# Sidebar for filtering
st.sidebar.header("Filter Data")
start_date = st.sidebar.date_input("Start Date", value=data.index.min())
end_date = st.sidebar.date_input("End Date", value=data.index.max())

filtered_data = data[(data.index >= pd.Timestamp(start_date)) & (data.index <= pd.Timestamp(end_date))]

# Filter by product
product_filter = st.sidebar.multiselect("Select Products", data['nama_produk'].unique(), default=data['nama_produk'].unique())
filtered_data = filtered_data[filtered_data['nama_produk'].isin(product_filter)]

# Main Dashboard
st.title("Dashboard Penjualan dengan Prediksi")
st.write("Data penjualan yang ditampilkan berdasarkan filter yang dipilih.")

st.dataframe(filtered_data)

# Visualize data
st.subheader("Jumlah Penjualan per Tanggal")
fig, ax = plt.subplots()
for product in product_filter:
    product_data = filtered_data[filtered_data['nama_produk'] == product]
    ax.plot(product_data.index, product_data['jumlah'], label=f"{product} (Aktual)")

ax.set_xlabel("Tanggal")
ax.set_ylabel("Jumlah Penjualan")
ax.legend()
st.pyplot(fig)

# Prediksi Moving Average
st.subheader("Prediksi Penjualan (Moving Average dan ARIMA)")
window_size = st.slider("Window Size (Moving Average)", min_value=3, max_value=14, value=7)

predictions = pd.DataFrame()
for product in product_filter:
    product_data = filtered_data[filtered_data['nama_produk'] == product].resample('D').sum()
    product_data['ma'] = product_data['jumlah'].rolling(window=window_size).mean()
    
    # Prediksi dengan ARIMA jika data cukup
    if len(product_data.dropna()) > 10:
        model = ARIMA(product_data['jumlah'].dropna(), order=(5, 1, 0))
        arima_result = model.fit()
        future_forecast = arima_result.forecast(steps=30)
        product_data['prediksi_arima'] = future_forecast
        ax.plot(future_forecast.index, future_forecast, '--', label=f"{product} (ARIMA Prediksi)")
    
    ax.plot(product_data.index, product_data['ma'], label=f"{product} (Moving Average)")

ax.legend()
st.pyplot(fig)
