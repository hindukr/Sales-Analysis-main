import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Advanced Sales Dashboard", layout="wide")

st.title("🚀 Regional Sales Analytics Dashboard")


# =========================================================
# 🎨 n8n STYLE DARK UI (ADDED ONLY)
# =========================================================
st.markdown("""
<style>

/* 🌑 Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: #e2e8f0;
}

/* Title */
h1 {
    color: #f8fafc;
    text-align: center;
    font-weight: 700;
}

/* Headers */
h2, h3 {
    color: #38bdf8;
}

/* KPI Cards */
[data-testid="metric-container"] {
    background: #020617;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 15px;
}

/* Charts */
.stPlotlyChart {
    background: #020617;
    border-radius: 10px;
    border: 1px solid #1e293b;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #020617;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    color: white;
    border-radius: 8px;
    border: none;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #020617;
    border: 1px solid #1e293b;
    border-radius: 10px;
}

/* Text */
p, label {
    color: #cbd5f5 !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# FILE UPLOAD
# -----------------------------
file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

if file:

    # Load file
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.write("### 📄 Data Preview")
    st.dataframe(df.head())

    # -----------------------------
    # AUTO COLUMN DETECTION
    # -----------------------------
    def find_col(names):
        for n in names:
            for c in df.columns:
                if n in c.lower():
                    return c
        return None

    date_col = find_col(["date"])
    revenue_col = find_col(["revenue", "sales", "total"])
    profit_col = find_col(["profit", "margin"])
    region_col = find_col(["region"])
    product_col = find_col(["product"])
    customer_col = find_col(["customer"])

    # -----------------------------
    # CLEANING
    # -----------------------------
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df['Year'] = df[date_col].dt.year
        df['Month'] = df[date_col].dt.month

    df.fillna(df.median(numeric_only=True), inplace=True)

    # -----------------------------
    # SIDEBAR FILTERS
    # -----------------------------
    st.sidebar.header("🎯 Filters")

    if region_col:
        region_filter = st.sidebar.multiselect(
            "Select Region", df[region_col].unique(), default=df[region_col].unique()
        )
        df = df[df[region_col].isin(region_filter)]

    if product_col:
        product_filter = st.sidebar.multiselect(
            "Select Product", df[product_col].unique(), default=df[product_col].unique()
        )
        df = df[df[product_col].isin(product_filter)]

    if date_col:
        date_range = st.sidebar.date_input(
            "Select Date Range",
            [df[date_col].min(), df[date_col].max()]
        )
        if len(date_range) == 2:
            df = df[(df[date_col] >= pd.to_datetime(date_range[0])) &
                    (df[date_col] <= pd.to_datetime(date_range[1]))]

    # -----------------------------
    # KPIs
    # -----------------------------
    st.subheader("📊 Key Metrics")

    col1, col2, col3 = st.columns(3)

    if revenue_col:
        col1.metric("Total Revenue", f"{df[revenue_col].sum():,.0f}")

    if profit_col:
        col2.metric("Total Profit", f"{df[profit_col].sum():,.0f}")

    col3.metric("Records", len(df))

    # -----------------------------
    # INTERACTIVE CHARTS
    # -----------------------------
    st.subheader("📈 Interactive Analysis")

    if region_col and revenue_col:
        fig = px.bar(df.groupby(region_col)[revenue_col].sum().reset_index(),
                     x=region_col, y=revenue_col, title="Revenue by Region")
        st.plotly_chart(fig)

    if product_col and revenue_col:
        fig = px.pie(df, names=product_col, values=revenue_col,
                     title="Product Contribution")
        st.plotly_chart(fig)

    if date_col and revenue_col:
        trend = df.groupby(df[date_col].dt.to_period("M"))[revenue_col].sum().reset_index()
        trend[date_col] = trend[date_col].astype(str)

        fig = px.line(trend, x=date_col, y=revenue_col, title="Monthly Trend")
        st.plotly_chart(fig)

    # -----------------------------
    # CORRELATION
    # -----------------------------
    st.subheader("🔗 Correlation")

    num_df = df.select_dtypes(include=np.number)

    if len(num_df.columns) > 1:
        fig = px.imshow(num_df.corr(), text_auto=True, title="Correlation Heatmap")
        st.plotly_chart(fig)

    # -----------------------------
    # CUSTOMER SEGMENTATION
    # -----------------------------
    st.subheader("🧠 Customer Segmentation")

    if customer_col and revenue_col and profit_col:
        seg = df.groupby(customer_col)[[revenue_col, profit_col]].sum()

        scaler = StandardScaler()
        scaled = scaler.fit_transform(seg)

        kmeans = KMeans(n_clusters=3, random_state=42)
        seg['Cluster'] = kmeans.fit_predict(scaled)

        fig = px.scatter(seg,
                         x=revenue_col,
                         y=profit_col,
                         color=seg['Cluster'].astype(str),
                         title="Customer Segments")
        st.plotly_chart(fig)

    # -----------------------------
    # SALES PREDICTION (ML)
    # -----------------------------
    st.subheader("📈 Sales Prediction")

    if date_col and revenue_col:
        df_model = df[[date_col, revenue_col]].dropna()
        df_model = df_model.sort_values(date_col)

        # Convert date to number
        df_model['Days'] = (df_model[date_col] - df_model[date_col].min()).dt.days

        X = df_model[['Days']]
        y = df_model[revenue_col]

        model = LinearRegression()
        model.fit(X, y)

        future_days = np.arange(X.max()[0], X.max()[0] + 30).reshape(-1, 1)
        preds = model.predict(future_days)

        future_dates = pd.date_range(start=df_model[date_col].max(), periods=30)

        pred_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Sales": preds
        })

        fig = px.line(pred_df, x="Date", y="Predicted Sales",
                      title="Next 30 Days Sales Prediction")
        st.plotly_chart(fig)

else:
    st.info("👆 Upload a dataset to start")

# =========================================================
# 🚀 ADVANCED INSIGHTS 
# =========================================================

if file:

    st.subheader("🔮 Prediction Output + Recommendations")

    if date_col and revenue_col:

        df_model = df[[date_col, revenue_col]].dropna().sort_values(date_col)

        if len(df_model) > 5:

            # Convert date to numeric
            df_model['Days'] = (df_model[date_col] - df_model[date_col].min()).dt.days

            from sklearn.linear_model import LinearRegression
            model = LinearRegression()

            X = df_model[['Days']]
            y = df_model[revenue_col]

            model.fit(X, y)

            # Predict next 30 days
            future_days = np.arange(X.max()[0], X.max()[0] + 30).reshape(-1, 1)
            preds = model.predict(future_days)

            future_dates = pd.date_range(
                start=df_model[date_col].max(),
                periods=30
            )

            pred_df = pd.DataFrame({
                "Date": future_dates,
                "Predicted Revenue": preds
            })

            # -----------------------------
            # 📊 SHOW PREDICTION OUTPUT
            # -----------------------------
            st.write("### 📈 Predicted Revenue (Next 30 Days)")
            st.dataframe(pred_df.head())

            # -----------------------------
            # 📊 PREDICTION GRAPH
            # -----------------------------
            import plotly.express as px

            fig = px.line(pred_df, x="Date", y="Predicted Revenue",
                          title="Future Sales Prediction")
            st.plotly_chart(fig)

            # -----------------------------
            # 🧠 BUSINESS RECOMMENDATIONS
            # -----------------------------
            st.write("### 💡 Smart Business Recommendations")

            avg_growth = (preds[-1] - preds[0]) / preds[0]

            recommendations = []

            # Growth logic
            if avg_growth > 0.1:
                recommendations.append("📈 Sales are increasing — expand inventory and invest in marketing.")
            elif avg_growth < -0.05:
                recommendations.append("⚠️ Sales are declining — consider discounts or promotional campaigns.")
            else:
                recommendations.append("📊 Sales are stable — optimize pricing strategy.")

            # Region insights
            if region_col:
                top_region = df.groupby(region_col)[revenue_col].sum().idxmax()
                recommendations.append(f"🌍 Focus on high-performing region: {top_region}")

            # Product insights
            if product_col:
                top_product = df.groupby(product_col)[revenue_col].sum().idxmax()
                recommendations.append(f"🛍️ Promote top product: {top_product}")

            # Profit insight
            if profit_col:
                profit_ratio = df[profit_col].sum() / df[revenue_col].sum()
                if profit_ratio < 0.1:
                    recommendations.append("💰 Profit margin is low — reduce costs or increase pricing.")
                else:
                    recommendations.append("💹 Healthy profit margins — consider scaling operations.")

            # Show recommendations
            for rec in recommendations:
                st.write("-", rec)

        else:
            st.warning("Not enough data for prediction (minimum 5 records required)")

# =========================================================
# 🤖 AI EXPLANATION + 📄 PDF REPORT + 📈 ADVANCED FORECAST
# =========================================================

if file:

    st.subheader("🤖 AI Insights + Advanced Forecasting")

    import io

    # -----------------------------
    # 🤖 AI-STYLE TEXT EXPLANATION
    # -----------------------------
    if revenue_col:

        total_revenue = df[revenue_col].sum()
        avg_revenue = df[revenue_col].mean()

        insight_text = f"""
        The dataset shows a total revenue of {total_revenue:,.0f} with an average of {avg_revenue:,.2f}.
        Sales performance indicates that business growth is {'increasing' if avg_revenue > 0 else 'stable'}.
        Key contributing factors include product performance, regional demand, and pricing strategy.
        """

        if region_col:
            top_region = df.groupby(region_col)[revenue_col].sum().idxmax()
            insight_text += f"\nThe highest revenue is generated from {top_region}, indicating a strong market presence."

        if product_col:
            top_product = df.groupby(product_col)[revenue_col].sum().idxmax()
            insight_text += f"\nThe top-performing product is {top_product}, contributing significantly to total sales."

        st.write("### 🧠 AI Generated Business Summary")
        st.success(insight_text)

    # -----------------------------
    # 📄 PDF REPORT GENERATION
    # -----------------------------
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    def generate_pdf(text):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()

        content = []
        content.append(Paragraph("Sales Analysis Report", styles['Title']))
        content.append(Spacer(1, 12))
        content.append(Paragraph(text, styles['BodyText']))

        doc.build(content)
        buffer.seek(0)
        return buffer

    if revenue_col:
        pdf_buffer = generate_pdf(insight_text)

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_buffer,
            file_name="sales_report.pdf",
            mime="application/pdf"
        )

    # -----------------------------
    # 📈 ADVANCED FORECAST (POLY MODEL)
    # -----------------------------
    st.write("### 📈 Advanced Forecast (Improved Prediction)")

    if date_col and revenue_col:

        df_model = df[[date_col, revenue_col]].dropna().sort_values(date_col)

        if len(df_model) > 10:

            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression

            df_model['Days'] = (df_model[date_col] - df_model[date_col].min()).dt.days

            X = df_model[['Days']]
            y = df_model[revenue_col]

            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)

            model = LinearRegression()
            model.fit(X_poly, y)

            future_days = np.arange(X.max()[0], X.max()[0] + 30).reshape(-1, 1)
            future_poly = poly.transform(future_days)

            preds = model.predict(future_poly)

            future_dates = pd.date_range(
                start=df_model[date_col].max(),
                periods=30
            )

            forecast_df = pd.DataFrame({
                "Date": future_dates,
                "Forecast Revenue": preds
            })

            import plotly.express as px

            fig = px.line(forecast_df, x="Date", y="Forecast Revenue",
                          title="Advanced Sales Forecast")
            st.plotly_chart(fig)

        else:
            st.warning("Not enough data for advanced forecasting")



    
    

# =========================================================
# 📄 CONTINUOUS FLOW PDF (NO FORCED PAGE BREAKS)
# =========================================================

if file:

    st.subheader("📄 Download Flow Report")

    import io
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet
    import matplotlib.pyplot as plt

    def save_chart(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return buf

    def generate_flow_report():

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()

        content = []

        # -----------------------------
        # TITLE
        # -----------------------------
        content.append(Paragraph("Sales Analysis Report", styles['Title']))
        content.append(Spacer(1, 12))

        # -----------------------------
        # KPIs
        # -----------------------------
        content.append(Paragraph("Key Metrics", styles['Heading2']))

        content.append(Paragraph(f"Total Records: {len(df)}", styles['BodyText']))

        if revenue_col:
            content.append(Paragraph(f"Total Revenue: {df[revenue_col].sum():,.0f}", styles['BodyText']))

        if profit_col:
            content.append(Paragraph(f"Total Profit: {df[profit_col].sum():,.0f}", styles['BodyText']))

        content.append(Spacer(1, 12))

        # -----------------------------
        # CHARTS
        # -----------------------------
        content.append(Paragraph("Visual Analysis", styles['Heading2']))

        if revenue_col:
            fig, ax = plt.subplots()
            df[revenue_col].hist(ax=ax)
            ax.set_title("Revenue Distribution")
            content.append(Image(save_chart(fig), width=400, height=250))
            content.append(Spacer(1, 12))
            plt.close(fig)

        if region_col and revenue_col:
            fig, ax = plt.subplots()
            df.groupby(region_col)[revenue_col].sum().plot(kind='bar', ax=ax)
            ax.set_title("Revenue by Region")
            content.append(Image(save_chart(fig), width=400, height=250))
            content.append(Spacer(1, 12))
            plt.close(fig)

        # -----------------------------
        # AI SUMMARY (FIXED)
        # -----------------------------
        content.append(Paragraph("AI Business Summary", styles['Heading2']))

        ai_summary = ""

        if revenue_col:
            total = df[revenue_col].sum()
            avg = df[revenue_col].mean()

            trend = "growing" if avg > df[revenue_col].median() else "stable"

            ai_summary = f"""
            The overall sales performance is {trend}, with a total revenue of {total:,.0f}.
            The average revenue per transaction is {avg:,.2f}, indicating consistent business activity.
            """

            if region_col:
                top_region = df.groupby(region_col)[revenue_col].sum().idxmax()
                ai_summary += f" The region '{top_region}' is the top contributor to revenue."

            if product_col:
                top_product = df.groupby(product_col)[revenue_col].sum().idxmax()
                ai_summary += f" The product '{top_product}' shows strong demand in the market."

        else:
            ai_summary = "Not enough data available to generate insights."

        content.append(Paragraph(ai_summary, styles['BodyText']))
        content.append(Spacer(1, 12))

        # -----------------------------
        # RECOMMENDATIONS
        # -----------------------------
        content.append(Paragraph("Recommendations", styles['Heading2']))

        if revenue_col:
            content.append(Paragraph("- Sales are strong — consider expansion.", styles['BodyText']))

        if region_col:
            top_region = df.groupby(region_col)[revenue_col].sum().idxmax()
            content.append(Paragraph(f"- Focus on region: {top_region}", styles['BodyText']))

        if product_col:
            top_product = df.groupby(product_col)[revenue_col].sum().idxmax()
            content.append(Paragraph(f"- Promote product: {top_product}", styles['BodyText']))

        content.append(Spacer(1, 12))

        # -----------------------------
        # FORECAST
        # -----------------------------
        content.append(Paragraph("Forecast Insight", styles['Heading2']))
        content.append(Paragraph(
            "Sales forecasting indicates expected trends for the next 30 days based on historical data.",
            styles['BodyText']
        ))

        # Build PDF
        doc.build(content)
        buffer.seek(0)
        return buffer

    flow_pdf = generate_flow_report()

    st.download_button(
        label="📥 Download Continuous Flow Report",
        data=flow_pdf,
        file_name="smooth_sales_report.pdf",
        mime="application/pdf",
        key="flow_report_pdf"
    )