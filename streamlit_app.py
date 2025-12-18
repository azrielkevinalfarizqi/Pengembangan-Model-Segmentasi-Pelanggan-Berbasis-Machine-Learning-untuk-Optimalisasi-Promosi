import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots


# ======================= LOAD DATA UTAMA =======================

df = pd.read_csv(
    "data.csv",
    encoding="latin1",
    low_memory=False,
    dtype={
        "InvoiceNo": "string",
        "StockCode": "string",
        "Description": "string",
        "CustomerID": "string",
        "Country": "category",
        "R_Score": "int8",
        "F_Score": "int8",
        "M_Score": "int8",
        "RFM_Score": "int16",
    },
    parse_dates=["InvoiceDate"]
)

# ===== Pastikan numerik =====
num_cols = [
    "Quantity", "UnitPrice", "TotalAmount",
    "Recency", "Frequency", "Monetary",
    "Total_transaction"
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ===== Drop baris invalid =====
df = df.loc[
    df["CustomerID"].notna() &
    df["InvoiceNo"].notna() &
    df["TotalAmount"].notna()
]

# ===== Feature waktu (kalau belum ada) =====
df["InvoiceYearMonth"] = df["InvoiceDate"].dt.to_period("M")
df["InvoiceDate_only"] = df["InvoiceDate"].dt.date
df["DayName"] = df["InvoiceDate"].dt.day_name().astype("category")
df["Hour"] = df["InvoiceDate"].dt.hour
df["InvoiceMonthName"] = df["InvoiceDate"].dt.month_name().astype("category")

data = pd.read_csv( "customer_segmentation.csv", encoding="latin1", low_memory=False, dtype={ "CustomerID": "string", "RFM_Segment": "category", "Cluster": "int8", "R_Score": "int8", "F_Score": "int8", "M_Score": "int8" } ) 
num_cols = [ "Recency", "Frequency", "Monetary", "Quantity_total", "UnitPrice_avg", "TotalAmount_total", "Total_transaction_total", "InvoiceYearMonth_num", "RFM_Score" ] 
for col in num_cols: data[col] = pd.to_numeric(data[col], errors="coerce")

# ============================================================================================

# PAGE CONFIG
st.set_page_config(page_title="A25-CS313", layout="wide")

st.title("Customer Insight Mining: Pendekatan RFM dan Machine Learning untuk Meningkatkan Loyalitas Pelanggan")

# === TAB ===
tab_visualization, tab_rfm, tab_clustering, tab_insight = st.tabs(["VISUALISASI DATA AWAL",
                                                                   "RFM ANALYSIS",
                                                                   "CLUSTERING ANALYSIS",
                                                                   "INTERPRETASI"])

with tab_visualization:
#============ VISUALISASI PEMBELI BERDASARKAN NEGARA (ATLAS WORLD MAP) =============
    st.subheader("ANALISIS PENJUALAN DAN PENDAPATAN BERDASARKAN NEGARA")

    # --- Hitung revenue, transaksi, dll ---
    country_info = (
        df.groupby("Country")
        .agg(
            TotalRevenue=("TotalAmount", "sum"),
            TransactionCount=("InvoiceNo", "nunique")
        )
        .reset_index()
    )

    country_info["Purchased"] = 1  # indikator negara pembeli

    # --- ALL COUNTRY LIST dari Plotly (gapminder) ---
    world = px.data.gapminder()[["country"]].drop_duplicates()
    world.columns = ["Country"]

    # --- Merge: negara pembeli + negara yang tidak beli ---
    world_map = world.merge(country_info, on="Country", how="left")
    world_map["Purchased"] = world_map["Purchased"].fillna(0)

    # --- warna: negara beli = warna atlas, negara lain = abu muda ---
    world_map["ColorValue"] = world_map["Purchased"].astype(int)

    # --- CHOROPLETH ATLAS MAP ---
    fig_atlas = px.choropleth(
        world_map,
        locations="Country",
        locationmode="country names",
        color="ColorValue",
        hover_name="Country",
        hover_data={
            "TotalRevenue": ":,.0f",
            "TransactionCount": ":,",
            "Purchased": False,
            "ColorValue": False
        },
        color_continuous_scale=["#d3d3d3", "#ff9933"],  # abu → oranye atlas
    )

    # --- STYLE SEPERTI ATLAS ---
    fig_atlas.update_geos(
        showcountries=True,
        showcoastlines=True,
        showland=True,
        landcolor="white",
        oceancolor="#f8f8f8",
        lakecolor="#f8f8f8",
        projection_type="natural earth"
    )

    fig_atlas.update_layout(
        height=700,
        width=1500,
        coloraxis_showscale=False,   # sembunyikan legend warna
        title="Peta Persebaran Pelanggan Secara Global",
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(fig_atlas, use_container_width=True)



#======== TOTAL PEMASUKAN PER NEGARA ============
    with st.expander("Penjualan Berdasarkan Negara"):
        # Grouping
        country = (
            df.groupby('Country')
            .agg(TotalRevenue=('TotalAmount', 'sum'),
                TransactionCount=('TotalAmount', 'count'),
                TotalQuantity=('Quantity', 'sum'),
                UniqueInvoices=('InvoiceNo', 'nunique'))
            .sort_values('TotalRevenue', ascending=False)
        )

        # Persentase pemasukan
        country['RevenuePercentage'] = (
            country['TotalRevenue'] / country['TotalRevenue'].sum() * 100
        ).round(2)

        # Ambil 5 besar
        top = country.head(5).reset_index()   # <--- PENTING: Country tetap "Country"

        # Palet warna
        PALETTE = [
            "#FF8C00", "#FFA733", "#FFA726", "#FFB74D", "#FFBE66",
            "#FFCC80", "#FFD599", "#FFECCC", "#FFF5E6", "#FFE0B2"
        ]

        # Barchart
        fig = px.bar(
            top,
            x="Country",
            y="TotalRevenue",
            text="TotalRevenue",
            color="Country",
            color_discrete_sequence=PALETTE,
            title="5 Negara dengan Penjualan Terbesar"
        )

        # Format Hover + Teks (HASIL PERSENTASE BELUMMM JELAS DAN JELEK MAKANYA DIHAPUS)
        fig.update_traces(
            texttemplate='£%{y:,.0f}',
            textposition='outside',
            hovertemplate=
                "<b>%{x}</b><br>" +
                "Revenue: £%{y:,.0f}<br>" 
        )

        # Tampilkan chart
        st.plotly_chart(fig, use_container_width=True)

        # Info Negara Terbesar
        top_country = top.iloc[0]["Country"]
        pct = top.iloc[0]["RevenuePercentage"]
        rev = top.iloc[0]["TotalRevenue"]

        st.success(
            f"**Negara dengan Penjualan terbesar: `{top_country}`**\n"
            f"- Total Pemasukan: **£{rev:,.0f}**\n"
            # f"- Share: **{pct:.2f}%**"
        )

    #======== TOTAL PEMASUKAN PER NEGARA (EXCLUDE UK) ============
    with st.expander("Penjualan Berdasarkan Negara (Tanpa UK)"):
        # Filter negara
        df_filtered = df[df['Country'] != 'United Kingdom'].copy()

        # Hitung total amount
        df_filtered['TotalAmount'] = df_filtered['Quantity'] * df_filtered['UnitPrice']

        # Grouping per negara
        country = (
            df_filtered.groupby('Country')
                .agg(
                    TotalRevenue=('TotalAmount', 'sum'),
                    TransactionCount=('TotalAmount', 'count'),
                    TotalQuantity=('Quantity', 'sum'),
                    UniqueInvoices=('InvoiceNo', 'nunique')
                )
                .sort_values('TotalRevenue', ascending=False)
        )

        # Persentase revenue
        country['RevenuePercentage'] = (
            country['TotalRevenue'] / country['TotalRevenue'].sum() * 100
        ).round(2)

        # Ambil 10 teratas
        top = country.head(10).reset_index()

        # Palet warna
        PALETTE = [
            "#FF8C00", "#FFA733", "#FFA726", "#FFB74D", "#FFBE66",
            "#FFCC80", "#FFD599", "#FFECCC", "#FFF5E6", "#FFE0B2"
        ]

        # Barchart
        fig = px.bar(
            top,
            x="Country",
            y="TotalRevenue",
            text="TotalRevenue",
            color="Country",
            color_discrete_sequence=PALETTE,
            title="Top 10 Negara (Tanpa UK)"
        )

        fig.update_traces(
            texttemplate='£%{y:,.0f}',
            textposition='outside',
            hovertemplate=
                "<b>%{x}</b><br>" +
                "Revenue: £%{y:,.0f}<br>"
        )

        fig.update_layout(
            xaxis_tickangle=-45
        )

        st.plotly_chart(fig, use_container_width=True)

        # Insight negara teratas
        top_country = top.iloc[0]["Country"]
        pct = top.iloc[0]["RevenuePercentage"]
        rev = top.iloc[0]["TotalRevenue"]

        st.success(
            f"**Negara dengan penjualan terbesar (tanpa UK): `{top_country}`**\n"
            f"- Total Pemasukan: **£{rev:,.0f}**"
        )

#======== NEGARA DENGAN PENJUALAN PALING SEDIKIT ============
    with st.expander("Negara dengan Penjualan Paling Sedikit"):
        # Ambil 10 negara terbawah berdasarkan TotalRevenue
        bottom = (
            country
            .sort_values('TotalRevenue', ascending=True)
            .head(5)
            .reset_index()
        )

        # Barchart
        fig_bottom = px.bar(
            bottom,
            x="Country",
            y="TotalRevenue",
            text="TotalRevenue",
            color="Country",
            color_discrete_sequence=PALETTE,  # boleh pakai palet warna yang sama
            title="5 Negara dengan Penjualan Paling Sedikit"
        )

        fig_bottom.update_traces(
            texttemplate='£%{y:,.0f}',
            textposition='outside',
            hovertemplate=
                "<b>%{x}</b><br>" +
                "Revenue: £%{y:,.0f}<br>"
        )

        st.plotly_chart(fig_bottom, use_container_width=True)

        # Insight negara dengan penjualan paling sedikit
        low_country = bottom.iloc[0]["Country"]
        low_rev = bottom.iloc[0]["TotalRevenue"]
        low_pct = bottom.iloc[0]["RevenuePercentage"]

        st.info(
            f"**Negara dengan penjualan paling sedikit: `{low_country}`**\n"
            f"- Total pemasukan: **£{low_rev:,.0f}**\n"
            # f"- Share: **{low_pct:.2f}%**"
        )

# ==================== Tren Pendapatan Bulanan =======================
    with st.expander("Tren Pendapatan Bulanan Tahun 2011-2012"):
        monthly = (
            df.groupby('InvoiceYearMonth')
            .agg(
                TotalAmount=('TotalAmount', 'sum'),
                Orders=('InvoiceNo', 'nunique'),
                Active_Customers=('CustomerID', 'nunique')
            )
            .reset_index()
        )

        # Konversi ke string agar tampil rapi di plot
        monthly['InvoiceYearMonth'] = monthly['InvoiceYearMonth'].astype(str)

        # Hitung Average Order Value (AOV)
        monthly['AOV'] = monthly['TotalAmount'] / monthly['Orders']

        # Warna garis
        LINE_COLOR = "#FF8C00"

        # Plotly line chart
        fig_monthly = px.line(
            monthly,
            x="InvoiceYearMonth",
            y="TotalAmount",
            markers=True,
            title="Tren Pendapatan Bulanan Tahun 2011-2012",
        )

        fig_monthly.update_traces(
            line=dict(width=3, color=LINE_COLOR),
            marker=dict(size=8),
            hovertemplate=
                "<b>%{x}</b><br>" +
                "Total Amount: £%{y:,.0f}<br>" +
                "Orders: %{customdata[0]:,}<br>" +
                "Active Customers: %{customdata[1]:,}<br>" +
                "AOV: £%{customdata[2]:,.2f}<extra></extra>",
            customdata=monthly[['Orders', 'Active_Customers', 'AOV']].values
        )

        fig_monthly.update_layout(
            xaxis_title="Month",
            yaxis_title="Total Amount (£)",
            xaxis_tickangle=-45,
            plot_bgcolor="white",
            height=450,
        )

        # Tampilkan chart di Streamlit
        st.plotly_chart(fig_monthly, use_container_width=True)

        # ============ Insight otomatis ============
        best_month = monthly.loc[monthly['TotalAmount'].idxmax()]
        worst_month = monthly.loc[monthly['TotalAmount'].idxmin()]

        insight = f"""
        - **Bulan dengan transaksi (Total Amount) tertinggi:** `{best_month['InvoiceYearMonth']}`  
        Total: **£{best_month['TotalAmount']:,.0f}**

        - **Bulan dengan transaksi terendah:** `{worst_month['InvoiceYearMonth']}`  
        Total: **£{worst_month['TotalAmount']:,.0f}**

        - **Rata-rata AOV keseluruhan:** £{monthly['AOV'].mean():,.2f}
        """

        st.info(insight)

#======== MONTHLY TREND BY COUNTRY ============
    with st.expander("Tren Pendapatan Bulanan Berdasarkan Negara"):
        
        # Dropdown negara
        selected_country = st.selectbox(
            "Pilih Negara:",
            sorted(df['Country'].unique()),
            key="selected_country_monthly"
        )

        # Filter data untuk negara terpilih
        df_country = df[df['Country'] == selected_country].copy()

        # Pastikan TotalAmount sudah ada (jaga-jaga)
        df_country['TotalAmount'] = df_country['Quantity'] * df_country['UnitPrice']

        # Aggregasi bulanan
        monthly_cty = (
            df_country.groupby('InvoiceYearMonth')
            .agg(
                TotalAmount=('TotalAmount', 'sum'),
                Orders=('InvoiceNo', 'nunique'),
                Active_Customers=('CustomerID', 'nunique')
            )
            .reset_index()
            .sort_values('InvoiceYearMonth')
        )

        # Ubah ke string agar tampil rapi
        monthly_cty['InvoiceYearMonth'] = monthly_cty['InvoiceYearMonth'].astype(str)

        # Hitung AOV
        monthly_cty['AOV'] = monthly_cty['TotalAmount'] / monthly_cty['Orders']

        # Plot
        fig_cty = px.line(
            monthly_cty,
            x="InvoiceYearMonth",
            y="TotalAmount",
            markers=True,
            title=f"Tren Pendapatan Bulanan – {selected_country}",
        )

        fig_cty.update_traces(
            line=dict(width=3, color="#FF8C00"),
            marker=dict(size=8),
            hovertemplate=
                "<b>%{x}</b><br>" +
                "Total Amount: £%{y:,.0f}<br>" +
                "Orders: %{customdata[0]:,}<br>" +
                "Active Customers: %{customdata[1]:,}<br>" +
                "AOV: £%{customdata[2]:,.2f}<extra></extra>",
            customdata=monthly_cty[['Orders', 'Active_Customers', 'AOV']].values
        )

        fig_cty.update_layout(
            xaxis_title="Month",
            yaxis_title="Total Amount (£)",
            xaxis_tickangle=-45,
            plot_bgcolor="white",
            height=450,
        )

        st.plotly_chart(fig_cty, use_container_width=True)

        #============= INSIGHT OTOMATIS =============
        if len(monthly_cty) > 0:
            best_m = monthly_cty.loc[monthly_cty['TotalAmount'].idxmax()]
            worst_m = monthly_cty.loc[monthly_cty['TotalAmount'].idxmin()]
            
            insight_cty = f"""
            - **Bulan dengan pendapatan tertinggi:** `{best_m['InvoiceYearMonth']}`  
            Total: **£{best_m['TotalAmount']:,.0f}**

            - **Bulan dengan pendapatan terendah:** `{worst_m['InvoiceYearMonth']}`  
            Total: **£{worst_m['TotalAmount']:,.0f}**

            - **Rata-rata AOV negara `{selected_country}`:** £{monthly_cty['AOV'].mean():,.2f}
            """

            st.info(insight_cty)
        else:
            st.warning("Tidak ada data untuk negara ini.")


#======== PENJUALAN PRODUK BERDASARKAN REVENUE ============
    st.subheader("ANALISIS PENJUALAN DAN PENDAPATAN BERDASARKAN PRODUK")
    with st.expander("Penjualan Produk Berdasarkan Revenue"):
    # --- Agregasi revenue per produk ---
        product = (
            df.groupby('Description')
            .agg(
                TotalRevenue=('TotalAmount', 'sum'),
                UniqueInvoices=('InvoiceNo', 'nunique'),
                AvgPrice=('UnitPrice', 'mean')
            )
            .reset_index()
        )

        # Sort berdasarkan revenue terbesar
        product = product.sort_values('TotalRevenue', ascending=False)

        # Ambil top 10
        top_prod = product.head(10)

        # Warna palet
        PALETTE = [
            "#FF8C00", "#FFA733", "#FFA726", "#FFB74D", "#FFBE66",
            "#FFCC80", "#FFD599", "#FFECCC", "#FFF5E6", "#FFE0B2"
        ]

        # --- Barchart Total Revenue ---
        fig_prod = px.bar(
            top_prod,
            x="Description",
            y="TotalRevenue",
            text="TotalRevenue",
            color="Description",
            color_discrete_sequence=PALETTE,
            title="Top 10 Produk dengan Revenue Tertinggi"
        )

        fig_prod.update_traces(
            texttemplate='£%{y:,.0f}',
            textposition='outside',
            hovertemplate=
                "<b>%{x}</b><br>" +
                "Total Revenue: £%{y:,.0f}<br>" +
                "Avg Price: £%{customdata[0]:.2f}<extra></extra>",
            customdata=top_prod[['AvgPrice']].values
        )

        fig_prod.update_layout(
            xaxis_title="Product",
            yaxis_title="Total Revenue (£)",
            xaxis_tickangle=-45,
            showlegend=False
        )

        st.plotly_chart(fig_prod, use_container_width=True)

        # --- Insight ---
        top_name = top_prod.iloc[0]['Description']
        top_rev = top_prod.iloc[0]['TotalRevenue']

        summary = f"""
        - **Produk dengan Revenue Tertinggi:** `{top_name}`
        - **Total Revenue:** £{top_rev:,.0f}
        """

        st.info(summary)

#======== PENJUALAN PRODUK BERDASARKAN QUANTITY ============
    with st.expander("Penjualan Produk Berdasarkan Jumlah Produk Terjual"):
        # --- Hitung total quantity per produk ---
        product_qty = (
            df.groupby('Description')['Quantity']
            .sum()
            .reset_index()
            .sort_values('Quantity', ascending=False)
        )

        # Ambil top 10
        top_qty = product_qty.head(10)

        # Warna palet
        PALETTE = [
            "#FF8C00", "#FFA733", "#FFA726", "#FFB74D", "#FFBE66",
            "#FFCC80", "#FFD599", "#FFECCC", "#FFF5E6", "#FFE0B2"
        ]

        # --- Barchart Quantity Terjual ---
        fig_qty = px.bar(
            top_qty,
            x="Description",
            y="Quantity",
            text="Quantity",
            color="Description",
            color_discrete_sequence=PALETTE,
            title="Top 10 Produk Berdasarkan Jumlah Quantity Terjual"
        )

        fig_qty.update_traces(
            texttemplate='%{y:,}',
            textposition='outside',
            hovertemplate=
                "<b>%{x}</b><br>" +
                "Quantity Terjual: %{y:,}<extra></extra>"
        )

        fig_qty.update_layout(
            xaxis_title="Product",
            yaxis_title="Quantity Sold",
            xaxis_tickangle=-45,
            showlegend=False
        )

        st.plotly_chart(fig_qty, use_container_width=True)

        # --- Insight ---
        top_name = top_qty.iloc[0]['Description']
        top_q = top_qty.iloc[0]['Quantity']

        summary = f"""
        - **Produk dengan Quantity Terjual Terbanyak:** `{top_name}`
        - **Total Quantity Terjual:** {top_q:,}
        """

        st.info(summary)

#======== SCATTER PLOT: REVENUE vs QUANTITY (ALL PRODUCTS) ============
    with st.expander("Persebaran Penjualan Produk Berdasarkan Pendapatan dan Jumlah Produk Terjual"):
        # --- Buat agregasi revenue & quantity per produk ---
        product_scatter = (
            df.groupby('Description')
            .agg(
                TotalRevenue=('TotalAmount', 'sum'),
                TotalQuantity=('Quantity', 'sum'),
                AvgPrice=('UnitPrice', 'mean')
            )
            .reset_index()
        )
        
        product_scatter = product_scatter[product_scatter["TotalRevenue"] > 0]

        # --- Scatter Plot ---
        fig_scatter = px.scatter(
            product_scatter,
            x="TotalQuantity",
            y="TotalRevenue",
            hover_name="Description",
            hover_data={
                "TotalQuantity": True,
                "TotalRevenue": ":,.0f",
                "AvgPrice": ":,.2f"
            },
            title="Scatter Plot: Total Revenue vs Quantity per Product",
        )

        fig_scatter.update_layout(
            xaxis_title="Total Quantity Sold",
            yaxis_title="Total Revenue (£)",
            height=600,
            plot_bgcolor="white"
        )

        fig_scatter.update_traces(
            marker=dict(opacity=0.7, line=dict(width=1, color="black"))
        )

        # --- Tampilkan chart ---
        st.plotly_chart(fig_scatter, use_container_width=True)

#======== ANALISIS AKTIVITAS PELANGGAN ============
    st.subheader("ANALISIS AKTIVITAS PELANGGAN")

#======== ANALISIS AKTIVITAS PELANGGAN PER HARI ============   
    with st.expander("Keaktifan Pelanggan Berdasarkan Hari"):
        # Urutan hari (biar tidak acak)
        order_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        # Grouping count transaksi per hari
        day_sales = (
            df.groupby('DayName')
            .agg(TransactionCount=('InvoiceNo', 'nunique'))
            .reindex(order_days)   # pastikan urut
            .reset_index()
        )
        # Palet warna
        PALETTE = [
            "#FF8C00", "#FFA733", "#FFA726", "#FFB74D",
            "#FFBE66", "#FFCC80", "#FFD599"
        ]

        # Barchart
        fig = px.bar(
            day_sales,
            x="DayName",
            y="TransactionCount",
            text="TransactionCount",
            color="DayName",
            color_discrete_sequence=PALETTE,
            title="Jumlah Transaksi Pelanggan Berdasarkan Hari"
        )

        # Hover + Label
        fig.update_traces(
            textposition='outside',
            hovertemplate=
                "<b>%{x}</b><br>" +
                "Transaksi: %{y:,}<extra></extra>"
        )

        # Layout
        fig.update_layout(
            xaxis_title="Hari",
            yaxis_title="Jumlah Transaksi",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # Insight
        top_day = day_sales.loc[day_sales['TransactionCount'].idxmax()]
        st.success(
            f"**Hari dengan transaksi terbanyak: `{top_day['DayName']}`**\n"
            f"- Jumlah Transaksi: **{top_day['TransactionCount']:,}**"
        )
#======== ANALISIS AKTIVITAS PER JAM BERDASARKAN HARI ============   
    with st.expander("Keaktifan Pelanggan Berdasarkan Jam & Hari"):

        # Dropdown hari (Bahasa Indonesia)
        order_days = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]

        selected_day = st.selectbox(
            "Pilih Hari:",
            order_days,
            key="selected_day_hour"
        )

        # Mapping ke DayName Bahasa Inggris di dataset
        day_map = {
            "Senin": "Monday",
            "Selasa": "Tuesday",
            "Rabu": "Wednesday",
            "Kamis": "Thursday",
            "Jumat": "Friday",
            "Sabtu": "Saturday",
            "Minggu": "Sunday"
        }

        # Filter sesuai hari (SETELAH mapping)
        df_day = df[df['DayName'] == day_map[selected_day]]

        # Group per jam
        hourly_sales = (
            df_day.groupby("Hour")
            .agg(TransactionCount=('InvoiceNo', 'nunique'))
            .reset_index()
        )

        # Pastikan jam 0–23 muncul semua
        all_hours = pd.DataFrame({"Hour": range(24)})
        hourly_sales = all_hours.merge(hourly_sales, on="Hour", how="left").fillna(0)

        # Line chart
        fig_hour = px.line(
            hourly_sales,
            x="Hour",
            y="TransactionCount",
            markers=True,
            title=f"Trend Jumlah Transaksi per Jam – {selected_day}"
        )

        fig_hour.update_traces(
            line=dict(width=3, color="#FF8C00"),
            marker=dict(size=8),
            hovertemplate="<b>Jam %{x}:00</b><br>Transaksi: %{y:,}<extra></extra>"
        )

        fig_hour.update_layout(
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=1
            ),
            xaxis_title="Jam",
            yaxis_title="Jumlah Transaksi",
            plot_bgcolor="white",
            height=450
        )

        st.plotly_chart(fig_hour, use_container_width=True)

        # Insight
        top_hour = hourly_sales.loc[hourly_sales['TransactionCount'].idxmax()]
        st.success(
            f"**Jam paling aktif pada hari {selected_day}: pukul {int(top_hour['Hour'])}:00**\n"
            f"- Total transaksi: **{int(top_hour['TransactionCount']):,}**"
        )
#======== ANALISIS AKTIVITAS PELANGGAN PER BULAN ============   
    with st.expander("Keaktifan Pelanggan Berdasarkan Bulan"):

        # Urutan bulan agar rapi
        order_months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]

        # Grouping count transaksi per bulan
        month_sales = (
            df.groupby('InvoiceMonthName')
            .agg(TransactionCount=('InvoiceNo', 'nunique'))
            .reindex(order_months)     # agar urut
            .reset_index()
        )

        # Warna
        PALETTE = [
            "#FF8C00", "#FFA733", "#FFA726", "#FFB74D",
            "#FFBE66", "#FFCC80", "#FFD599", "#FFE0B2",
            "#FFECCC", "#FFF5E6", "#FFE5CC", "#FFD8B2"
        ]

        # Barchart
        fig_month = px.bar(
            month_sales,
            x="InvoiceMonthName",
            y="TransactionCount",
            text="TransactionCount",
            color="InvoiceMonthName",
            color_discrete_sequence=PALETTE,
            title="Jumlah Transaksi Pelanggan Berdasarkan Bulan"
        )

        # Hover + Label
        fig_month.update_traces(
            textposition='outside',
            hovertemplate=
                "<b>%{x}</b><br>" +
                "Transaksi: %{y:,}<extra></extra>"
        )

        # Layout
        fig_month.update_layout(
            xaxis_title="Bulan",
            yaxis_title="Jumlah Transaksi",
            showlegend=False,
            xaxis_tickangle=-45
        )

        st.plotly_chart(fig_month, use_container_width=True)

        # Insight
        top_month = month_sales.loc[month_sales['TransactionCount'].idxmax()]
        st.success(
            f"**Bulan dengan transaksi terbanyak: `{top_month['InvoiceMonthName']}`**\n"
            f"- Jumlah Transaksi: **{top_month['TransactionCount']:,}**"
        )

#======== TAB RFM ANALYSIS ============ 
with tab_rfm:
    st.subheader("ANALISIS PELANGGAN BERDASARKAN RFM SEGMENTATION")
    #======== Pelanggan Berdasarkan RFM Segmentation ============   
    with st.expander("Distribusi Pelanggan Berdasarkan RFM Segmentation"):

        # Hitung jumlah customer per segmen
        segment_counts = (
            df.dropna(subset=["RFM_Segment"])
            .groupby("RFM_Segment")["CustomerID"]
            .nunique()
            .reset_index(name="Count")
        )
        segment_counts.columns = ["RFM_Segment", "Count"]

        # Urutkan berdasarkan jumlah terbanyak
        segment_counts = segment_counts.sort_values("Count", ascending=False)
        total_customers = len(data)

        # Hitung persentase
        segment_counts["Percentage"] = (segment_counts["Count"] / total_customers) * 100

        # Barchart dengan Plotly
        fig_segment = px.bar(
            segment_counts,
            x="RFM_Segment",
            y="Count",
            text=segment_counts["Percentage"].apply(lambda x: f"{x:.1f}%"),
            color="RFM_Segment",
            color_discrete_sequence=px.colors.qualitative.Set3,
            title="Distribusi Customer per RFM Segment (%)"
        )

        fig_segment.update_traces(
            textposition="outside",
            hovertemplate=
                "<b>Segment: %{x}</b><br>" +
                "Jumlah Customer: %{y}<br>" +
                "Persentase: %{text}<extra></extra>"
        )

        fig_segment.update_layout(
            xaxis_title="RFM Segment",
            yaxis_title="Jumlah Customer",
            xaxis_tickangle=45,
            showlegend=False
        )

        st.plotly_chart(fig_segment, use_container_width=True)

        # INSIGHT
        top_segment = segment_counts.iloc[0]
        st.success(
            f"**Segmen dengan jumlah pelanggan terbanyak: `{top_segment['RFM_Segment']}`**\n"
            f"- Jumlah customer: **{top_segment['Count']:,}**\n"
            f"- Persentase: **{top_segment['Percentage']:.1f}%**"
        )

    # ========== RADAR CHART RFM SCORE PER SEGMENT ==========
    with st.expander("Radar Chart RFM Score Berdasarkan Segmen Customer"):

        # Pastikan kolom score numerik
        for col in ["R_Score", "F_Score", "M_Score"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Dropdown segment
        selected_segment = st.selectbox(
            "Pilih Segment RFM:",
            sorted(df["RFM_Segment"].dropna().unique()),
            key="selected_segment_radar"
        )

        # Filter data sesuai segment
        seg_data = df[df["RFM_Segment"] == selected_segment].copy()

        # Hitung rata-rata score
        avg_r = seg_data["R_Score"].mean()
        avg_f = seg_data["F_Score"].mean()
        avg_m = seg_data["M_Score"].mean()

        # Data radar
        radar_df = pd.DataFrame({
            "Metric": ["Recency Score", "Frequency Score", "Monetary Score"],
            "Value": [avg_r, avg_f, avg_m]
        })

        # Tutup radar (kembali ke titik awal)
        radar_df = pd.concat([radar_df, radar_df.iloc[[0]]])

        # ===== PLOT RADAR =====
        fig_radar = go.Figure()

        fig_radar.add_trace(go.Scatterpolar(
            r=radar_df["Value"],
            theta=radar_df["Metric"],
            fill="toself",
            name=selected_segment,
            line=dict(width=3),
            hovertemplate="<b>%{theta}</b>: %{r:.2f}<extra></extra>"
        ))

        fig_radar.update_layout(
            title=f"RFM Score Radar – {selected_segment}",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5],
                    showline=True,
                    linewidth=1,
                    gridcolor="lightgray"
                )
            ),
            height=500,
            showlegend=False
        )

        st.plotly_chart(fig_radar, use_container_width=True)

        # ========= INSIGHT OTOMATIS ==========
        st.subheader("Insight Segment Otomatis")

        insight = f"""
        Segment: `{selected_segment}`

        - Recency Score (R): {avg_r:.2f}  
        - Frequency Score (F): {avg_f:.2f}  
        - Monetary Score (M): {avg_m:.2f}  

        Interpretasi:
        """

        if avg_r >= 4:
            insight += "- Pelanggan sangat recent aktif.\n"
        elif avg_r >= 3:
            insight += "- Aktivitas pelanggan cukup recent.\n"
        else:
            insight += "- Pelanggan lama tidak bertransaksi.\n"

        if avg_f >= 4:
            insight += "- Memiliki frekuensi pembelian tinggi.\n"
        elif avg_f >= 3:
            insight += "- Frekuensi pembelian menengah.\n"
        else:
            insight += "- Frekuensi pembelian rendah.\n"

        if avg_m >= 4:
            insight += "- Memberikan kontribusi pendapatan besar.\n"
        elif avg_m >= 3:
            insight += "- Kontribusi pendapatan menengah.\n"
        else:
            insight += "- Kontribusi pendapatan rendah.\n"

        st.info(insight)

    #======== PIE CHART: PROPORSI REVENUE PER RFM SEGMENT ============
    with st.expander("Proporsi Revenue per RFM Segment"):

        # Pastikan numerik
        df["TotalAmount"] = pd.to_numeric(
            df["TotalAmount"], errors="coerce"
        )

        # Agregasi revenue per segment
        segment_revenue = (
            df.groupby("RFM_Segment")
            .agg(TotalRevenue=("TotalAmount", "sum"))
            .reset_index()
        )

        # Hitung persentase
        total_revenue = segment_revenue["TotalRevenue"].sum()
        segment_revenue["Percentage"] = (
            segment_revenue["TotalRevenue"] / total_revenue * 100
        )

        # Pie / Donut chart
        fig_pie = px.pie(
            segment_revenue,
            names="RFM_Segment",
            values="TotalRevenue",
            hole=0.45,  # donut style
            title="Proporsi Revenue Berdasarkan RFM Segment"
        )

        fig_pie.update_traces(
            textinfo="percent+label",
            hovertemplate=
                "<b>%{label}</b><br>" +
                "Revenue: £%{value:,.0f}<br>" +
                "Persentase: %{percent}<extra></extra>"
        )

        fig_pie.update_layout(
            height=500
        )

        st.plotly_chart(fig_pie, use_container_width=True)

        # ===== INSIGHT OTOMATIS =====
        top_segment = segment_revenue.loc[
            segment_revenue["TotalRevenue"].idxmax()
        ]

        st.success(
            f"**Segment dengan kontribusi revenue terbesar: `{top_segment['RFM_Segment']}`**\n"
            f"- Total Revenue: **£{top_segment['TotalRevenue']:,.0f}**\n"
            f"- Kontribusi: **{top_segment['Percentage']:.1f}%**"
        )

    #======== AOV PER RFM SEGMENT ============
    with st.expander("Average Order Value (AOV) per RFM Segment"):

        # ===== Pastikan tipe data =====
        df["TotalAmount"] = pd.to_numeric(df["TotalAmount"], errors="coerce")
        df["CustomerID"] = df["CustomerID"].astype(str)
        df["InvoiceNo"] = df["InvoiceNo"].astype(str)

        # ===== AOV per Customer =====
        aov_customer = (
            df
            .dropna(subset=["RFM_Segment"])
            .groupby(["CustomerID", "RFM_Segment"])
            .agg(
                TotalRevenue=("TotalAmount", "sum"),
                TotalOrders=("InvoiceNo", "nunique")
            )
            .reset_index()
        )

        aov_customer["AOV"] = (
            aov_customer["TotalRevenue"] / aov_customer["TotalOrders"]
        )

        # ===== AOV per Segment =====
        aov_segment = (
            aov_customer
            .groupby("RFM_Segment")
            .agg(
                Avg_AOV=("AOV", "mean"),
                Customer_Count=("CustomerID", "nunique")
            )
            .reset_index()
            .sort_values("Avg_AOV", ascending=False)
        )

        if aov_segment.empty:
            st.warning("Data AOV tidak tersedia.")
        else:
            fig_aov = px.bar(
                aov_segment,
                x="RFM_Segment",
                y="Avg_AOV",
                color="RFM_Segment",
                text=aov_segment["Avg_AOV"].apply(lambda x: f"£{x:,.2f}"),
                custom_data=["Customer_Count"],
                color_discrete_sequence=px.colors.qualitative.Set3,
                title="Average Order Value (AOV) per RFM Segment"
            )

            fig_aov.update_traces(
                textposition="outside",
                hovertemplate=
                    "<b>%{x}</b><br>" +
                    "Average AOV: £%{y:,.2f}<br>" +
                    "Jumlah Customer: %{customdata[0]:,}<extra></extra>"
            )

            fig_aov.update_layout(
                xaxis_title="RFM Segment",
                yaxis_title="Average Order Value (£)",
                showlegend=False,
                height=450
            )

            st.plotly_chart(fig_aov, use_container_width=True)

            # ===== Insight otomatis =====
            top_seg = aov_segment.iloc[0]
            low_seg = aov_segment.iloc[-1]

            st.info(
                f"""
                **Insight AOV per Segment:**
                - Segment dengan **AOV tertinggi**: `{top_seg['RFM_Segment']}`  
                Rata-rata AOV: **£{top_seg['Avg_AOV']:,.2f}**

                - Segment dengan **AOV terendah**: `{low_seg['RFM_Segment']}`  
                Rata-rata AOV: **£{low_seg['Avg_AOV']:,.2f}**
                """
            )

    #======== TOP 5 NEGARA PER RFM SEGMENT ============
    with st.expander("Top 5 Negara Berdasarkan RFM Segment"):

        # Dropdown RFM Segment
        selected_segment = st.selectbox(
            "Pilih RFM Segment:",
            sorted(df["RFM_Segment"].dropna().unique()),
            key="selected_rfm_segment_country"
        )

        # Filter data sesuai segment
        seg_country = df[df["RFM_Segment"] == selected_segment]

        # ===== Agregasi: jumlah customer unik per negara =====
        country_segment = (
            seg_country
            .groupby("Country")
            .agg(
                Customer_Count=("CustomerID", "nunique")
            )
            .reset_index()
            .sort_values("Customer_Count", ascending=False)
            .head(5)
        )

        # ===== Bar chart =====
        fig_country = px.bar(
            country_segment,
            x="Country",
            y="Customer_Count",
            text="Customer_Count",
            color="Country",
            color_discrete_sequence=px.colors.qualitative.Set3,
            title=f"Top 5 Negara – Segment {selected_segment}"
        )

        fig_country.update_traces(
            textposition="outside",
            hovertemplate=
                "<b>%{x}</b><br>" +
                "Jumlah Customer: %{y:,}<extra></extra>"
        )

        fig_country.update_layout(
            xaxis_title="Negara",
            yaxis_title="Jumlah Customer Unik",
            showlegend=False,
            height=450
        )

        st.plotly_chart(fig_country, use_container_width=True)

        # ===== INSIGHT OTOMATIS =====
        if not country_segment.empty:
            top_country = country_segment.iloc[0]

            st.success(
                f"""
                **Insight Segment `{selected_segment}`**
                - Negara dengan jumlah customer terbanyak: **{top_country['Country']}**
                - Total customer: **{int(top_country['Customer_Count']):,}**
                """
            )
        else:
            st.warning("Tidak ada data untuk segment ini.")

    #======== TOP 5 PRODUK PER RFM SEGMENT ============
    with st.expander("Top 5 Produk Berdasarkan RFM Segment"):

        # ===== Pastikan tipe data =====
        df["CustomerID"] = df["CustomerID"].astype(str)
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
        df["TotalAmount"] = pd.to_numeric(df["TotalAmount"], errors="coerce")

        # ===== Dropdown Segment =====
        selected_segment = st.selectbox(
            "Pilih RFM Segment:",
            sorted(df["RFM_Segment"].dropna().unique()),
            key="selected_rfm_segment_product"
        )

        # ===== Dropdown Metric =====
        metric_option = st.selectbox(
            "Pilih Dasar Ranking Produk:",
            ["Revenue", "Quantity"],
            key="selected_product_metric"
        )

        # ===== Filter data sesuai segment =====
        seg_product = df[df["RFM_Segment"] == selected_segment]

        # ===== Agregasi produk =====
        if metric_option == "Revenue":
            product_segment = (
                seg_product
                .groupby("Description")
                .agg(
                    TotalRevenue=("TotalAmount", "sum")
                )
                .reset_index()
                .sort_values("TotalRevenue", ascending=False)
                .head(5)
            )

            y_col = "TotalRevenue"
            y_label = "Total Revenue (£)"
            title = f"Top 5 Produk – Segment {selected_segment} (by Revenue)"
            text_format = lambda x: f"£{x:,.0f}"

        else:  # Quantity
            product_segment = (
                seg_product
                .groupby("Description")
                .agg(
                    TotalQuantity=("Quantity", "sum")
                )
                .reset_index()
                .sort_values("TotalQuantity", ascending=False)
                .head(5)
            )

            y_col = "TotalQuantity"
            y_label = "Total Quantity"
            title = f"Top 5 Produk – Segment {selected_segment} (by Quantity)"
            text_format = lambda x: f"{int(x):,}"

        # ===== Bar chart =====
        fig_product = px.bar(
            product_segment,
            x="Description",
            y=y_col,
            text=product_segment[y_col].apply(text_format),
            color="Description",
            color_discrete_sequence=px.colors.qualitative.Set3,
            title=title
        )

        fig_product.update_traces(
            textposition="outside",
            hovertemplate=
                "<b>%{x}</b><br>" +
                f"{y_label}: %{{y:,}}<extra></extra>"
        )

        fig_product.update_layout(
            xaxis_title="Produk",
            yaxis_title=y_label,
            showlegend=False,
            height=500,
            xaxis_tickangle=-45
        )

        st.plotly_chart(fig_product, use_container_width=True)

        # ===== INSIGHT OTOMATIS =====
        if not product_segment.empty:
            top_product = product_segment.iloc[0]

            st.success(
                f"""
                **Insight Segment `{selected_segment}`**
                - Produk utama pendorong segment ini: **{top_product['Description']}**
                - Berdasarkan **{metric_option}**
                """
            )
        else:
            st.warning("Tidak ada data produk untuk segment ini.")

#======== TAB CLUSTERING ANALYSIS ============ 
with tab_clustering:
    st.subheader("ANALISIS PELANGGAN BERDASARKAN CLUSTERING SEGMENTATION")
    #======== DISTRIBUSI CUSTOMER PER CLUSTER ============
    with st.expander("Distribusi Pelanggan Berdasarkan Cluster"):
        # ===== Hitung jumlah customer unik per cluster =====
        cluster_dist = (
            data
            .groupby("Cluster")["CustomerID"]
            .nunique()
            .reset_index(name="Customer_Count")
            .sort_values("Customer_Count", ascending=False)
        )

        # ===== Bar Chart =====
        fig_cluster = px.bar(
            cluster_dist,
            x="Cluster",
            y="Customer_Count",
            text="Customer_Count",
            color="Cluster",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Distribusi Customer Berdasarkan Cluster"
        )

        fig_cluster.update_traces(
            textposition="outside",
            hovertemplate=
                "<b>Cluster %{x}</b><br>" +
                "Jumlah Customer: %{y:,}<extra></extra>"
        )

        fig_cluster.update_layout(
            xaxis_title="Cluster",
            yaxis_title="Jumlah Customer Unik",
            showlegend=False,
            height=450
        )

        st.plotly_chart(fig_cluster, use_container_width=True)

        # ===== INSIGHT OTOMATIS =====
        top_cluster = cluster_dist.iloc[0]

        st.success(
            f"""
            **Insight Distribusi Cluster**
            - Cluster dengan jumlah customer terbanyak: **Cluster {top_cluster['Cluster']}**
            - Total customer: **{int(top_cluster['Customer_Count']):,}**
            """
        )

    #======== GROUPED BAR CHART NILAI RFM ASLI PER SCORE ============
    with st.expander("Distribusi Nilai RFM Asli Berdasarkan Score per Cluster"):

        # ===== Pastikan numerik =====
        for col in ["Recency", "Frequency", "Monetary", "R_Score", "F_Score", "M_Score"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        # ===== Dropdown Cluster =====
        selected_cluster = st.selectbox(
            "Pilih Cluster:",
            sorted(data["Cluster"].dropna().unique()),
            key="cluster_rfm_raw_score"
        )

        # ===== Filter cluster =====
        df_cluster = data[data["Cluster"] == selected_cluster].copy()

        plot_rows = []

        # ===== Loop score 1–5 =====
        for score in range(1, 6):

            r_val = df_cluster[df_cluster["R_Score"] == score]["Recency"].count()
            f_val = df_cluster[df_cluster["F_Score"] == score]["Frequency"].count()
            m_val = df_cluster[df_cluster["M_Score"] == score]["Monetary"].count()

            plot_rows.extend([
                {
                    "Score": f"Score {score}",
                    "Metric": "Recency",
                    "Value": r_val,
                    "Total_R": r_val,
                    "Total_F": f_val,
                    "Total_M": m_val
                },
                {
                    "Score": f"Score {score}",
                    "Metric": "Frequency",
                    "Value": f_val,
                    "Total_R": r_val,
                    "Total_F": f_val,
                    "Total_M": m_val
                },
                {
                    "Score": f"Score {score}",
                    "Metric": "Monetary",
                    "Value": m_val,
                    "Total_R": r_val,
                    "Total_F": f_val,
                    "Total_M": m_val
                },
            ])

        plot_df = pd.DataFrame(plot_rows)

        # ===== BAR CHART =====
        fig = px.bar(
            plot_df,
            x="Score",
            y="Value",
            color="Metric",
            barmode="group",
            custom_data=["Total_R", "Total_F", "Total_M"],
            title=f"Distribusi Nilai RFM Asli per Score – Cluster {selected_cluster}",
            color_discrete_map={
                "Recency": "#1f77b4",
                "Frequency": "#ff7f0e",
                "Monetary": "#2ca02c"
            }
        )

        # ===== HOVER TOTAL PER SCORE =====
        fig.update_traces(
            hovertemplate=
                "<b>%{x}</b><br><br>" +
                "Total Recency   : %{customdata[0]:,.0f}<br>" +
                "Total Frequency : %{customdata[1]:,.0f}<br>" +
                "Total Monetary  : %{customdata[2]:,.0f}" +
                "<extra></extra>"
        )

        # ===== SKALA Y DIPERKECIL (LOG SCALE) =====
        fig.update_layout(
            xaxis_title="Score RFM",
            yaxis_title="Total Nilai RFM Asli (Log Scale)",
            yaxis_type="log",
            yaxis=dict(
                showgrid=True,          # grid utama tetap ada
                gridcolor="rgba(200,200,200,0.6)",
                minor=dict(
                    showgrid=False      
                )
            ),

            plot_bgcolor="white",
            height=520
        )

        st.plotly_chart(fig, use_container_width=True)

    #======== PIE CHART: PROPORSI REVENUE PER CLUSTER ============
    with st.expander("Proporsi Revenue per Cluster"):
        # ===== Agregasi revenue per cluster =====
        cluster_revenue = (
            data
            .dropna(subset=["Cluster"])
            .groupby("Cluster")
            .agg(TotalRevenue=("TotalAmount_total", "sum"))
            .reset_index()
        )

        # ===== Hitung persentase =====
        total_revenue = cluster_revenue["TotalRevenue"].sum()
        cluster_revenue["Percentage"] = (
            cluster_revenue["TotalRevenue"] / total_revenue * 100
        )

        # ===== Pie / Donut chart =====
        fig_pie = px.pie(
            cluster_revenue,
            names="Cluster",
            values="TotalRevenue",
            hole=0.45,  # donut style
            title="Proporsi Revenue Berdasarkan Cluster"
        )

        fig_pie.update_traces(
            textinfo="percent+label",
            hovertemplate=
                "<b>Cluster %{label}</b><br>" +
                "Revenue: £%{value:,.0f}<br>" +
                "Persentase: %{percent}<extra></extra>"
        )

        fig_pie.update_layout(
            height=500
        )

        st.plotly_chart(fig_pie, use_container_width=True)

        # ===== INSIGHT OTOMATIS =====
        top_cluster = cluster_revenue.loc[
            cluster_revenue["TotalRevenue"].idxmax()
        ]

        st.success(
            f"""
            **Cluster dengan kontribusi revenue terbesar: Cluster {int(top_cluster['Cluster'])}**
            - Total Revenue: **£{top_cluster['TotalRevenue']:,.0f}**
            - Kontribusi: **{top_cluster['Percentage']:.1f}%**
            """
        )

    #======== BAR CHART: TOTAL QUANTITY PER CLUSTER ============
    with st.expander("Total Quantity per Cluster"):
        # ===== Agregasi total quantity per cluster =====
        cluster_quantity = (
            data
            .dropna(subset=["Cluster"])
            .groupby("Cluster")
            .agg(
                TotalQuantity=("Quantity_total", "sum")
            )
            .reset_index()
            .sort_values("TotalQuantity", ascending=False)
        )

        if cluster_quantity.empty:
            st.warning("Data quantity tidak tersedia.")
        else:
            # ===== BAR CHART =====
            fig_qty = px.bar(
                cluster_quantity,
                x="Cluster",
                y="TotalQuantity",
                text=cluster_quantity["TotalQuantity"].apply(lambda x: f"{int(x):,}"),
                title="Total Quantity Berdasarkan Cluster",
                color="Cluster",
                color_discrete_sequence=px.colors.qualitative.Set3
            )

            fig_qty.update_traces(
                textposition="outside",
                hovertemplate=
                    "<b>Cluster %{x}</b><br>" +
                    "Total Quantity: %{y:,}<extra></extra>"
            )

            fig_qty.update_layout(
                xaxis_title="Cluster",
                yaxis_title="Total Quantity",
                plot_bgcolor="white",
                height=450,
                showlegend=False
            )

            st.plotly_chart(fig_qty, use_container_width=True)

            # ===== INSIGHT OTOMATIS =====
            top_cluster = cluster_quantity.iloc[0]

            st.success(
                f"""
                **Cluster dengan total quantity tertinggi: Cluster {int(top_cluster['Cluster'])}**
                - Total Quantity: **{int(top_cluster['TotalQuantity']):,}**
                """
            )

   # ================= SCATTER PLOT VALIDASI CLUSTER =================
    with st.expander("Scatter Plot Antar Fitur (Per Cluster)"):

        axis_option = st.selectbox(
            "Pilih Kombinasi Sumbu:",
            [
                "Monetary vs Recency",
                "Monetary vs Frequency",
                "Frequency vs Recency"
            ],
            key="axis_option_cluster_scatter"  # 🔑 WAJIB UNIK
        )

        if axis_option == "Monetary vs Recency":
            x_col, y_col = "Recency", "Monetary"
        elif axis_option == "Monetary vs Frequency":
            x_col, y_col = "Frequency", "Monetary"
        else:
            x_col, y_col = "Recency", "Frequency"

        # ===== WARNA CLUSTER (FIXED) =====
        cluster_colors = {
            0: "#D61355",  # merah
            1: "#F94A29",  # oranye / kuning
            2: "#3EC70B",  # hijau
            3: "#1F6AE1"   # biru
        }

        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            color="Cluster",
            color_discrete_map=cluster_colors,
            opacity=0.75,
            title=f"Scatter Plot {y_col} vs {x_col}",
            hover_data=["CustomerID", "Recency", "Frequency", "Monetary"]
        )

        # ===== LOG SCALE KHUSUS MONETARY =====
        if x_col == "Monetary":
            fig.update_xaxes(type="log", title="Monetary (log scale)")
        if y_col == "Monetary":
            fig.update_yaxes(type="log", title="Monetary (log scale)")

        fig.update_layout(
            plot_bgcolor="white",
            height=550,
            legend_title_text="Cluster"
        )

        fig.update_traces(
            marker=dict(size=7),
            hovertemplate=
                "<b>CustomerID:</b> %{customdata[0]}<br>" +
                f"<b>{x_col}:</b> %{{x:,.0f}}<br>" +
                f"<b>{y_col}:</b> %{{y:,.0f}}<br>" +
                "<b>Cluster:</b> %{marker.color}<extra></extra>"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ========== STACKED BAR: RFM SEGMENT vs CLUSTER ==========
    with st.expander("Distribusi RFM Segment dalam Cluster"):

        # ===== Dropdown Cluster =====
        selected_cluster = st.selectbox(
            "Pilih Cluster:",
            sorted(data["Cluster"].dropna().unique())
        )

        # ===== Filter cluster =====
        cluster_data = (
            data[data["Cluster"] == selected_cluster]
            .dropna(subset=["RFM_Segment"])
            .groupby("RFM_Segment")
            .agg(Customer_Count=("CustomerID", "nunique"))
            .reset_index()
            .sort_values("Customer_Count", ascending=False)
        )

        # ===== Hitung persentase =====
        total_customer = cluster_data["Customer_Count"].sum()
        cluster_data["Percentage"] = (
            cluster_data["Customer_Count"] / total_customer * 100
        )

        # ===== Stacked Bar (single cluster → segment composition) =====
        fig_stack = px.bar(
            cluster_data,
            x="RFM_Segment",
            y="Customer_Count",
            text=cluster_data["Percentage"].apply(lambda x: f"{x:.1f}%"),
            color="RFM_Segment",
            title=f"Komposisi RFM Segment pada Cluster {selected_cluster}"
        )

        fig_stack.update_traces(
            textposition="outside",
            hovertemplate=
                "<b>%{x}</b><br>" +
                "Jumlah Customer: %{y:,}<br>" +
                "Proporsi: %{text}<extra></extra>"
        )

        fig_stack.update_layout(
            xaxis_title="RFM Segment",
            yaxis_title="Jumlah Customer",
            showlegend=False,
            height=450
        )

        st.plotly_chart(fig_stack, use_container_width=True)

        # ===== Insight Otomatis =====
        dominant_segment = cluster_data.iloc[0]

        st.info(
            f"""
            **Insight Cluster {selected_cluster}:**
            - Didominasi oleh segment **{dominant_segment['RFM_Segment']}**
            - Proporsi: **{dominant_segment['Percentage']:.1f}%**
            """
        )

#======== TAB INTERPRETASI ============ 
with tab_insight:
    # ================= SCATTER PLOT PER CLUSTER (DROPDOWN) =================
    with st.expander("Scatter Plot Antar Fitur"):

        # ===== Dropdown kombinasi axis =====
        axis_option = st.selectbox(
            "Pilih Kombinasi Sumbu:",
            [
                "Monetary vs Recency",
                "Monetary vs Frequency",
                "Frequency vs Recency"
            ],
            key="axis_scatter_per_cluster"
        )

        if axis_option == "Monetary vs Recency":
            x_col, y_col = "Recency", "Monetary"
        elif axis_option == "Monetary vs Frequency":
            x_col, y_col = "Frequency", "Monetary"
        else:
            x_col, y_col = "Recency", "Frequency"

        # ===== Dropdown Cluster =====
        selected_cluster = st.selectbox(
            "Pilih Cluster:",
            sorted(data["Cluster"].dropna().unique()),
            key="cluster_scatter_single"
        )

        # ===== Warna cluster (KONSISTEN) =====
        cluster_colors = {
            0: "#D61355",  # merah
            1: "#F94A29",  # oranye
            2: "#FCE22A",  # kuning
            3: "#30E3DF"   # biru
        }

        # ===== Filter data cluster =====
        df_cluster = data[data["Cluster"] == selected_cluster]

        # ===== Scatter Plot =====
        fig = px.scatter(
            df_cluster,
            x=x_col,
            y=y_col,
            color_discrete_sequence=[cluster_colors.get(selected_cluster, "#999999")],
            opacity=0.75,
            title=f"Scatter Plot {y_col} vs {x_col} — Cluster {selected_cluster}",
            hover_data=["CustomerID", "Recency", "Frequency", "Monetary"]
        )

        # ===== LOG SCALE KHUSUS MONETARY =====
        if x_col == "Monetary":
            fig.update_xaxes(type="log", title="Monetary (log scale)")
        if y_col == "Monetary":
            fig.update_yaxes(type="log", title="Monetary (log scale)")

        # ===== Layout & Grid =====
        fig.update_layout(
            plot_bgcolor="white",
            height=520,
            showlegend=False
        )

        # ===== MATIKAN MINOR GRID (BIAR RAPI) =====
        fig.update_yaxes(
            showgrid=True,
            gridcolor="lightgray",
            minor=dict(showgrid=False)
        )

        fig.update_xaxes(
            showgrid=True,
            gridcolor="lightgray",
            minor=dict(showgrid=False)
        )

        st.plotly_chart(fig, use_container_width=True)

        # ===== INSIGHT PER CLUSTER =====
        st.subheader("Insight Cluster")

        cluster_insights = {
        0: """
        *Cluster 0: High Value But At Risk*

        Cluster ini berisi pelanggan bernilai tinggi dengan intensitas transaksi yang kuat,
        namun menunjukkan indikasi penurunan aktivitas. Dominasi segmen Loyal Customer
        dan Can’t Lose Them menandakan risiko churn yang signifikan jika tidak dikelola
        secara proaktif. Strategi yang tepat adalah pendekatan personal, win-back campaign,
        dan penawaran eksklusif untuk mempertahankan nilai mereka.
        """,

        1: """
        *Cluster 1: VIP Customers*

        Cluster ini merepresentasikan pelanggan terbaik perusahaan yang didominasi segmen
        Champions. Mereka memiliki nilai transaksi tinggi, konsistensi pembelian yang stabil,
        dan loyalitas kuat. Fokus utama pada cluster ini adalah mempertahankan hubungan jangka
        panjang melalui program loyalitas, upselling, dan cross-selling.
        """,

        2: """
        *Cluster 2: Mass Customers*

        Cluster ini mencakup mayoritas pelanggan dengan nilai transaksi rendah hingga menengah.
        Banyak di antaranya berada pada fase tidak aktif atau berisiko churn. Meskipun kontribusi
        per pelanggan relatif kecil, cluster ini menjadi basis volume transaksi. Strategi yang
        disarankan adalah edukasi produk, promosi massal, dan reaktivasi ringan.
        """,

        3: """
        *Cluster 3: High Value Active*

        Cluster ini berisi pelanggan bernilai tinggi yang masih aktif dan stabil. Dominasi segmen
        Champions dan Loyal Customers menunjukkan potensi pertumbuhan lanjutan. Pelanggan di
        cluster ini ideal untuk strategi upselling, cross-selling, dan program loyalitas
        berjenjang guna meningkatkan lifetime value.
        """,
        }

        # Tampilkan insight sesuai cluster yang dipilih
        st.info(cluster_insights.get(selected_cluster, "Insight belum tersedia untuk cluster ini."))
        





