import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Diet Program Analytics Dashboard",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stAlert {
        background-color: #e8f4fd;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Google Drive URLs - Converted to direct download links
GOOGLE_DRIVE_URLS = {
    'customers': 'https://drive.google.com/file/d/1zJcsv6WFQVo1eNj78XwjJ1vC9fcUkqVr/view?usp=drive_link',
    'diet_programs': 'https://drive.google.com/file/d/19wrH-s98P_4yGHf5xeInAfhZUnfNe6ay/view?usp=drive_link',
    'customers_subscribers': 'https://drive.google.com/file/d/1vqxpwQTbh80GILVhrm-id6efxcXjn3mu/view?usp=drive_link',
    'selected_items': 'https://drive.google.com/file/d/1LBSRtbsYQ3Vmpgw7OaZ2cZWvH5RlyRs9/view?usp=drive_link'
}

# Load data with caching
@st.cache_data
def load_data():
    """Load all CSV files from Google Drive with error handling"""
    try:
        # Load customers data
        customers_df = pd.read_csv(GOOGLE_DRIVE_URLS['customers'])
        st.success("✅ Customers data loaded successfully from Google Drive")
        
        # Load diet programs data
        diet_programs_df = pd.read_csv(GOOGLE_DRIVE_URLS['diet_programs'])
        st.success("✅ Diet programs data loaded successfully from Google Drive")
        
        # Load customers with subscribers data
        customers_subscribers_df = pd.read_csv(GOOGLE_DRIVE_URLS['customers_subscribers'])
        st.success("✅ Customers subscribers data loaded successfully from Google Drive")
        
        # Load selected items data
        try:
            selected_items_df = pd.read_csv(GOOGLE_DRIVE_URLS['selected_items'])
            st.success("✅ Selected items data loaded successfully from Google Drive")
        except Exception as e:
            st.warning(f"⚠️ Could not load selected items data: {e}")
            selected_items_df = pd.DataFrame()
        
        return customers_df, diet_programs_df, customers_subscribers_df, selected_items_df
    
    except Exception as e:
        st.error(f"❌ Error loading data from Google Drive: {e}")
        return None, None, None, None

# Data preprocessing functions
@st.cache_data
def preprocess_customers(customers_df):
    """Preprocess customers data"""
    if customers_df is None:
        return pd.DataFrame()
    
    # Convert date columns
    date_columns = ['created_at', 'deleted_at']
    for col in date_columns:
        if col in customers_df.columns:
            customers_df[col] = pd.to_datetime(customers_df[col], errors='coerce')
    
    return customers_df

@st.cache_data
def preprocess_diet_programs(diet_programs_df):
    """Preprocess diet programs data"""
    if diet_programs_df is None:
        return pd.DataFrame()
    
    # Convert date columns
    date_columns = ['valid_from', 'valid_to', 'created_at', 'updated_at']
    for col in date_columns:
        if col in diet_programs_df.columns:
            diet_programs_df[col] = pd.to_datetime(diet_programs_df[col], errors='coerce')
    
    return diet_programs_df

@st.cache_data
def preprocess_subscribers(customers_subscribers_df):
    """Preprocess customers subscribers data"""
    if customers_subscribers_df is None:
        return pd.DataFrame()
    
    # Convert date columns
    date_columns = ['delivery_start_date', 'created_at_program']
    for col in date_columns:
        if col in customers_subscribers_df.columns:
            customers_subscribers_df[col] = pd.to_datetime(customers_subscribers_df[col], errors='coerce')
    
    return customers_subscribers_df

@st.cache_data
def preprocess_selected_items(selected_items_df):
    """Preprocess selected items data"""
    if selected_items_df is None or selected_items_df.empty:
        return pd.DataFrame()
    
    # Convert date columns
    if 'calender_date' in selected_items_df.columns:
        selected_items_df['calender_date'] = pd.to_datetime(selected_items_df['calender_date'], errors='coerce')
    
    return selected_items_df

# Load and preprocess data
customers_df, diet_programs_df, customers_subscribers_df, selected_items_df = load_data()

if customers_df is not None:
    customers_df = preprocess_customers(customers_df)
if diet_programs_df is not None:
    diet_programs_df = preprocess_diet_programs(diet_programs_df)
if customers_subscribers_df is not None:
    customers_subscribers_df = preprocess_subscribers(customers_subscribers_df)
if selected_items_df is not None:
    selected_items_df = preprocess_selected_items(selected_items_df)

# Sidebar navigation
st.sidebar.title("🥗 Diet Program Analytics")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["📊 Overview", "👥 Customer Analytics", "🍽️ Diet Programs", "📈 Subscriptions", "🍴 Meal Analysis", "🌍 Geographic Analysis"]
)

# Overview Page
if page == "📊 Overview":
    st.markdown('<h1 class="main-header">Diet Program Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    if customers_df is not None and not customers_df.empty:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_customers = len(customers_df)
            st.metric(
                label="Total Customers",
                value=f"{total_customers:,}",
                delta=f"+{len(customers_df[customers_df['created_at'] >= pd.Timestamp.now() - pd.Timedelta(days=30)]):,} this month"
            )
        
        with col2:
            if customers_subscribers_df is not None and not customers_subscribers_df.empty:
                active_subscriptions = len(customers_subscribers_df[customers_subscribers_df['status'] == 'Active'])
                total_subscriptions = len(customers_subscribers_df)
                st.metric(
                    label="Active Subscriptions",
                    value=f"{active_subscriptions:,}",
                    delta=f"{active_subscriptions/total_subscriptions*100:.1f}% of total"
                )
            else:
                st.metric(label="Active Subscriptions", value="N/A")
        
        with col3:
            if diet_programs_df is not None and not diet_programs_df.empty:
                total_programs = len(diet_programs_df)
                st.metric(
                    label="Diet Programs",
                    value=f"{total_programs}",
                    delta="Available programs"
                )
            else:
                st.metric(label="Diet Programs", value="N/A")
        
        with col4:
            if customers_df is not None and not customers_df.empty:
                total_nationalities = customers_df['nationality'].nunique()
                st.metric(
                    label="Nationalities",
                    value=f"{total_nationalities}",
                    delta="Countries served"
                )
            else:
                st.metric(label="Nationalities", value="N/A")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            if customers_df is not None and not customers_df.empty:
                # Customer growth over time
                customer_growth = customers_df.groupby(customers_df['created_at'].dt.to_period('M')).size().reset_index()
                customer_growth.columns = ['Month', 'New_Customers']
                customer_growth['Month'] = customer_growth['Month'].astype(str)
                
                fig_growth = px.line(
                    customer_growth, 
                    x='Month', 
                    y='New_Customers',
                    title='Customer Growth Over Time',
                    labels={'New_Customers': 'New Customers', 'Month': 'Month'}
                )
                fig_growth.update_layout(height=400)
                st.plotly_chart(fig_growth, use_container_width=True)
        
        with col2:
            if customers_subscribers_df is not None and not customers_subscribers_df.empty:
                # Program distribution
                program_dist = customers_subscribers_df['diet_program_name'].value_counts().head(10).reset_index()
                program_dist.columns = ['Program_Name', 'Count']
                
                fig_programs = px.pie(
                    program_dist,
                    values='Count',
                    names='Program_Name',
                    title='Top 10 Diet Program Subscriptions'
                )
                fig_programs.update_layout(height=400)
                st.plotly_chart(fig_programs, use_container_width=True)
        
        # Recent activity
        st.subheader("Recent Activity")
        if customers_df is not None and not customers_df.empty:
            recent_customers = customers_df.nlargest(10, 'created_at')[['username', 'created_at', 'nationality']]
            st.dataframe(recent_customers, use_container_width=True)

# Customer Analytics Page
elif page == "👥 Customer Analytics":
    st.title("Customer Analytics")
    
    if customers_df is not None and not customers_df.empty:
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            nationality_filter = st.multiselect(
                "Filter by Nationality",
                options=customers_df['nationality'].unique(),
                default=customers_df['nationality'].unique()
            )
        
        with col2:
            gender_filter = st.multiselect(
                "Filter by Gender",
                options=customers_df['gender'].unique(),
                default=customers_df['gender'].unique()
            )
        
        with col3:
            if 'created_at' in customers_df.columns:
                min_date = customers_df['created_at'].min()
                max_date = customers_df['created_at'].max()
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date.date(), max_date.date())
                )
            else:
                date_range = None
        
        # Apply filters
        filtered_customers = customers_df[
            (customers_df['nationality'].isin(nationality_filter)) &
            (customers_df['gender'].isin(gender_filter))
        ]
        
        if date_range and len(date_range) == 2:
            filtered_customers = filtered_customers[
                (filtered_customers['created_at'].dt.date >= date_range[0]) &
                (filtered_customers['created_at'].dt.date <= date_range[1])
            ]
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Customers by nationality
            nationality_counts = filtered_customers['nationality'].value_counts().head(15)
            
            fig_nationality = px.bar(
                x=nationality_counts.values,
                y=nationality_counts.index,
                orientation='h',
                title='Top 15 Customer Nationalities',
                labels={'x': 'Number of Customers', 'y': 'Nationality'}
            )
            fig_nationality.update_layout(height=500)
            st.plotly_chart(fig_nationality, use_container_width=True)
        
        with col2:
            # Customer gender distribution
            gender_counts = filtered_customers['gender'].value_counts()
            
            fig_gender = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title='Customer Gender Distribution'
            )
            fig_gender.update_layout(height=500)
            st.plotly_chart(fig_gender, use_container_width=True)
        
        # Customer demographics
        st.subheader("Customer Demographics")
        
        if 'age' in filtered_customers.columns:
            # Age distribution
            fig_age = px.histogram(
                filtered_customers,
                x='age',
                nbins=20,
                title='Customer Age Distribution',
                labels={'age': 'Age', 'y': 'Count'}
            )
            st.plotly_chart(fig_age, use_container_width=True)
        
        # BMI Analysis
        if 'bmi' in filtered_customers.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_bmi = px.histogram(
                    filtered_customers,
                    x='bmi',
                    nbins=20,
                    title='Customer BMI Distribution',
                    labels={'bmi': 'BMI', 'y': 'Count'}
                )
                st.plotly_chart(fig_bmi, use_container_width=True)
            
            with col2:
                # BMI by gender
                fig_bmi_gender = px.box(
                    filtered_customers,
                    x='gender',
                    y='bmi',
                    title='BMI Distribution by Gender'
                )
                st.plotly_chart(fig_bmi_gender, use_container_width=True)

# Diet Programs Page
elif page == "🍽️ Diet Programs":
    st.title("Diet Programs Analysis")
    
    if diet_programs_df is not None and not diet_programs_df.empty:
        # Program overview
        col1, col2 = st.columns(2)
        
        with col1:
            # Program popularity by master plan
            if 'master_plan_name' in diet_programs_df.columns:
                master_plan_counts = diet_programs_df['master_plan_name'].value_counts().head(10)
                
                fig_master_plans = px.bar(
                    x=master_plan_counts.values,
                    y=master_plan_counts.index,
                    orientation='h',
                    title='Top 10 Master Plans',
                    labels={'x': 'Number of Programs', 'y': 'Master Plan'}
                )
                fig_master_plans.update_layout(height=400)
                st.plotly_chart(fig_master_plans, use_container_width=True)
        
        with col2:
            # Calories distribution
            if 'calories_total' in diet_programs_df.columns:
                fig_calories = px.histogram(
                    diet_programs_df,
                    x='calories_total',
                    nbins=20,
                    title='Program Calories Distribution',
                    labels={'calories_total': 'Calories per Day', 'y': 'Count'}
                )
                fig_calories.update_layout(height=400)
                st.plotly_chart(fig_calories, use_container_width=True)
        
        # Program details table
        st.subheader("Program Details")
        
        # Filters for programs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'master_plan_name' in diet_programs_df.columns:
                master_plan_filter = st.multiselect(
                    "Filter by Master Plan",
                    options=diet_programs_df['master_plan_name'].unique(),
                    default=diet_programs_df['master_plan_name'].unique()
                )
            else:
                master_plan_filter = []
        
        with col2:
            if 'program_days' in diet_programs_df.columns:
                days_filter = st.multiselect(
                    "Filter by Program Days",
                    options=sorted(diet_programs_df['program_days'].unique()),
                    default=sorted(diet_programs_df['program_days'].unique())
                )
            else:
                days_filter = []
        
        with col3:
            if 'calories_total' in diet_programs_df.columns:
                min_calories = int(diet_programs_df['calories_total'].min())
                max_calories = int(diet_programs_df['calories_total'].max())
                calories_range = st.slider(
                    "Calories Range",
                    min_value=min_calories,
                    max_value=max_calories,
                    value=(min_calories, max_calories)
                )
            else:
                calories_range = None
        
        # Apply filters
        filtered_programs = diet_programs_df.copy()
        
        if master_plan_filter:
            filtered_programs = filtered_programs[filtered_programs['master_plan_name'].isin(master_plan_filter)]
        
        if days_filter:
            filtered_programs = filtered_programs[filtered_programs['program_days'].isin(days_filter)]
        
        if calories_range:
            filtered_programs = filtered_programs[
                (filtered_programs['calories_total'] >= calories_range[0]) &
                (filtered_programs['calories_total'] <= calories_range[1])
            ]
        
        # Display filtered programs
        display_columns = ['name_en_x', 'master_plan_name', 'calories_total', 'program_days', 'total_amount']
        available_columns = [col for col in display_columns if col in filtered_programs.columns]
        
        if available_columns:
            st.dataframe(filtered_programs[available_columns], use_container_width=True)
        
        # Program performance analysis
        st.subheader("Program Performance Analysis")
        
        if customers_subscribers_df is not None and not customers_subscribers_df.empty:
            # Merge with subscribers data
            program_performance = customers_subscribers_df.groupby('diet_program_name').agg({
                'customer_id': 'count',
                'paid_amount': 'sum',
                'status': lambda x: (x == 'Completed').sum()
            }).reset_index()
            
            program_performance.columns = ['Program_Name', 'Total_Subscriptions', 'Total_Revenue', 'Completed_Subscriptions']
            program_performance['Completion_Rate'] = (program_performance['Completed_Subscriptions'] / program_performance['Total_Subscriptions'] * 100).round(2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_subscriptions = px.bar(
                    program_performance.head(10),
                    x='Program_Name',
                    y='Total_Subscriptions',
                    title='Top 10 Programs by Subscriptions'
                )
                fig_subscriptions.update_xaxes(tickangle=45)
                st.plotly_chart(fig_subscriptions, use_container_width=True)
            
            with col2:
                fig_completion = px.bar(
                    program_performance.head(10),
                    x='Program_Name',
                    y='Completion_Rate',
                    title='Program Completion Rates (%)'
                )
                fig_completion.update_xaxes(tickangle=45)
                st.plotly_chart(fig_completion, use_container_width=True)

# Subscriptions Page
elif page == "📈 Subscriptions":
    st.title("Subscription Analytics")
    
    if customers_subscribers_df is not None and not customers_subscribers_df.empty:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_subscriptions = len(customers_subscribers_df)
            st.metric(
                label="Total Subscriptions",
                value=f"{total_subscriptions:,}",
                delta="All time"
            )
        
        with col2:
            active_subscriptions = len(customers_subscribers_df[customers_subscribers_df['status'] == 'Active'])
            st.metric(
                label="Active Subscriptions",
                value=f"{active_subscriptions:,}",
                delta=f"{active_subscriptions/total_subscriptions*100:.1f}%"
            )
        
        with col3:
            completed_subscriptions = len(customers_subscribers_df[customers_subscribers_df['status'] == 'Completed'])
            st.metric(
                label="Completed Subscriptions",
                value=f"{completed_subscriptions:,}",
                delta=f"{completed_subscriptions/total_subscriptions*100:.1f}%"
            )
        
        with col4:
            total_revenue = customers_subscribers_df['paid_amount'].sum()
            st.metric(
                label="Total Revenue",
                value=f"${total_revenue:,.0f}",
                delta="All time"
            )
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_filter = st.multiselect(
                "Filter by Status",
                options=customers_subscribers_df['status'].unique(),
                default=customers_subscribers_df['status'].unique()
            )
        
        with col2:
            program_filter = st.multiselect(
                "Filter by Program",
                options=customers_subscribers_df['diet_program_name'].unique(),
                default=customers_subscribers_df['diet_program_name'].unique()
            )
        
        with col3:
            if 'created_at_program' in customers_subscribers_df.columns:
                min_date = customers_subscribers_df['created_at_program'].min()
                max_date = customers_subscribers_df['created_at_program'].max()
                date_range = st.date_input(
                    "Subscription Date Range",
                    value=(min_date.date(), max_date.date())
                )
            else:
                date_range = None
        
        # Apply filters
        filtered_subscriptions = customers_subscribers_df[
            (customers_subscribers_df['status'].isin(status_filter)) &
            (customers_subscribers_df['diet_program_name'].isin(program_filter))
        ]
        
        if date_range and len(date_range) == 2:
            filtered_subscriptions = filtered_subscriptions[
                (filtered_subscriptions['created_at_program'].dt.date >= date_range[0]) &
                (filtered_subscriptions['created_at_program'].dt.date <= date_range[1])
            ]
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Subscription trends over time
            if 'created_at_program' in filtered_subscriptions.columns:
                subscription_trends = filtered_subscriptions.groupby(
                    filtered_subscriptions['created_at_program'].dt.to_period('M')
                ).size().reset_index()
                subscription_trends.columns = ['Month', 'Subscriptions']
                subscription_trends['Month'] = subscription_trends['Month'].astype(str)
                
                fig_trends = px.line(
                    subscription_trends,
                    x='Month',
                    y='Subscriptions',
                    title='Subscription Trends Over Time'
                )
                fig_trends.update_layout(height=400)
                st.plotly_chart(fig_trends, use_container_width=True)
        
        with col2:
            # Status distribution
            status_counts = filtered_subscriptions['status'].value_counts()
            
            fig_status = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title='Subscription Status Distribution'
            )
            fig_status.update_layout(height=400)
            st.plotly_chart(fig_status, use_container_width=True)
        
        # Revenue analysis
        st.subheader("Revenue Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue by program
            revenue_by_program = filtered_subscriptions.groupby('diet_program_name')['paid_amount'].sum().sort_values(ascending=False).head(10)
            
            fig_revenue = px.bar(
                x=revenue_by_program.values,
                y=revenue_by_program.index,
                orientation='h',
                title='Top 10 Programs by Revenue',
                labels={'x': 'Revenue ($)', 'y': 'Program'}
            )
            fig_revenue.update_layout(height=400)
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        with col2:
            # Average revenue by status
            avg_revenue_by_status = filtered_subscriptions.groupby('status')['paid_amount'].mean()
            
            fig_avg_revenue = px.bar(
                x=avg_revenue_by_status.index,
                y=avg_revenue_by_status.values,
                title='Average Revenue by Status',
                labels={'x': 'Status', 'y': 'Average Revenue ($)'}
            )
            fig_avg_revenue.update_layout(height=400)
            st.plotly_chart(fig_avg_revenue, use_container_width=True)

# Meal Analysis Page
elif page == "🍴 Meal Analysis":
    st.title("Meal and Item Analysis")
    
    if selected_items_df is not None and not selected_items_df.empty:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_items = len(selected_items_df)
            st.metric(
                label="Total Selected Items",
                value=f"{total_items:,}",
                delta="All time"
            )
        
        with col2:
            unique_items = selected_items_df['item_name'].nunique()
            st.metric(
                label="Unique Items",
                value=f"{unique_items:,}",
                delta="Available items"
            )
        
        with col3:
            unique_customers = selected_items_df['customer_id_y'].nunique()
            st.metric(
                label="Active Customers",
                value=f"{unique_customers:,}",
                delta="With meal selections"
            )
        
        with col4:
            if 'rate' in selected_items_df.columns:
                avg_rating = selected_items_df['rate'].mean()
                st.metric(
                    label="Average Rating",
                    value=f"{avg_rating:.2f}/5",
                    delta="Item ratings"
                )
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            meal_filter = st.multiselect(
                "Filter by Meal",
                options=selected_items_df['meal_name'].unique(),
                default=selected_items_df['meal_name'].unique()
            )
        
        with col2:
            program_filter = st.multiselect(
                "Filter by Program",
                options=selected_items_df['diet_program_name'].unique(),
                default=selected_items_df['diet_program_name'].unique()
            )
        
        with col3:
            if 'calender_date' in selected_items_df.columns:
                min_date = selected_items_df['calender_date'].min()
                max_date = selected_items_df['calender_date'].max()
                date_range = st.date_input(
                    "Meal Date Range",
                    value=(min_date.date(), max_date.date())
                )
            else:
                date_range = None
        
        # Apply filters
        filtered_items = selected_items_df[
            (selected_items_df['meal_name'].isin(meal_filter)) &
            (selected_items_df['diet_program_name'].isin(program_filter))
        ]
        
        if date_range and len(date_range) == 2:
            filtered_items = filtered_items[
                (filtered_items['calender_date'].dt.date >= date_range[0]) &
                (filtered_items['calender_date'].dt.date <= date_range[1])
            ]
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Most popular items
            popular_items = filtered_items['item_name'].value_counts().head(15)
            
            fig_popular = px.bar(
                x=popular_items.values,
                y=popular_items.index,
                orientation='h',
                title='Top 15 Most Popular Items',
                labels={'x': 'Selection Count', 'y': 'Item Name'}
            )
            fig_popular.update_layout(height=500)
            st.plotly_chart(fig_popular, use_container_width=True)
        
        with col2:
            # Meal distribution
            meal_counts = filtered_items['meal_name'].value_counts()
            
            fig_meals = px.pie(
                values=meal_counts.values,
                names=meal_counts.index,
                title='Meal Distribution'
            )
            fig_meals.update_layout(height=500)
            st.plotly_chart(fig_meals, use_container_width=True)
        
        # Rating analysis
        if 'rate' in filtered_items.columns:
            st.subheader("Item Rating Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Rating distribution
                fig_rating_dist = px.histogram(
                    filtered_items,
                    x='rate',
                    nbins=10,
                    title='Item Rating Distribution',
                    labels={'rate': 'Rating', 'y': 'Count'}
                )
                st.plotly_chart(fig_rating_dist, use_container_width=True)
            
            with col2:
                # Top rated items
                top_rated = filtered_items.groupby('item_name')['rate'].mean().sort_values(ascending=False).head(10)
                
                fig_top_rated = px.bar(
                    x=top_rated.values,
                    y=top_rated.index,
                    orientation='h',
                    title='Top 10 Rated Items',
                    labels={'x': 'Average Rating', 'y': 'Item Name'}
                )
                fig_top_rated.update_layout(height=400)
                st.plotly_chart(fig_top_rated, use_container_width=True)
        
        # Program day analysis
        if 'program_day' in filtered_items.columns:
            st.subheader("Program Day Analysis")
            
            day_counts = filtered_items['program_day'].value_counts().sort_index()
            
            fig_days = px.line(
                x=day_counts.index,
                y=day_counts.values,
                title='Item Selections by Program Day',
                labels={'x': 'Program Day', 'y': 'Number of Selections'}
            )
            st.plotly_chart(fig_days, use_container_width=True)

# Geographic Analysis Page
elif page == "🌍 Geographic Analysis":
    st.title("Geographic Analysis")
    
    if customers_df is not None and not customers_df.empty:
        # Geographic distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Top countries by customer count
            country_counts = customers_df['nationality'].value_counts().head(15)
            
            fig_countries = px.bar(
                x=country_counts.values,
                y=country_counts.index,
                orientation='h',
                title='Top 15 Countries by Customer Count',
                labels={'x': 'Number of Customers', 'y': 'Country'}
            )
            fig_countries.update_layout(height=600)
            st.plotly_chart(fig_countries, use_container_width=True)
        
        with col2:
            # Geographic heatmap (simulated)
            # Create a world map visualization
            world_data = pd.DataFrame({
                'Country': ['Kuwait', 'Egypt', 'Lebanon', 'Canada', 'Armenia', 'Algeria', 'Others'],
                'Customers': [
                    len(customers_df[customers_df['nationality'] == 'Kuwait']),
                    len(customers_df[customers_df['nationality'] == 'Egypt']),
                    len(customers_df[customers_df['nationality'] == 'Lebanon']),
                    len(customers_df[customers_df['nationality'] == 'Canada']),
                    len(customers_df[customers_df['nationality'] == 'Armenia']),
                    len(customers_df[customers_df['nationality'] == 'Algeria']),
                    len(customers_df[customers_df['nationality'] == 'Others'])
                ],
                'Latitude': [29.3117, 26.8206, 33.8547, 56.1304, 40.0691, 28.0339, 0],
                'Longitude': [47.4818, 30.8025, 35.8623, -106.3468, 45.0382, 1.6596, 0]
            })
            
            fig_map = px.scatter_mapbox(
                world_data,
                lat='Latitude',
                lon='Longitude',
                size='Customers',
                hover_name='Country',
                hover_data=['Customers'],
                title='Customer Distribution by Country',
                mapbox_style='carto-positron'
            )
            fig_map.update_layout(height=600)
            st.plotly_chart(fig_map, use_container_width=True)
        
        # Regional analysis
        st.subheader("Regional Analysis")
        
        # Merge with subscribers data for regional performance
        if customers_subscribers_df is not None and not customers_subscribers_df.empty:
            # Merge customers with subscribers
            regional_data = customers_subscribers_df.merge(
                customers_df[['id', 'nationality']], 
                left_on='customer_id', 
                right_on='id', 
                how='left'
            )

            # Use the correct column in groupby
            regional_performance = regional_data.groupby('nationality_x').agg({
                'customer_id': 'count',
                'paid_amount': 'sum',
                'status': lambda x: (x == 'Completed').sum()
            }).reset_index()
            
            regional_performance.columns = ['Country', 'Subscriptions', 'Revenue', 'Completed']
            regional_performance['Completion_Rate'] = (regional_performance['Completed'] / regional_performance['Subscriptions'] * 100).round(2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_regional_subscriptions = px.bar(
                    regional_performance.head(10),
                    x='Country',
                    y='Subscriptions',
                    title='Subscriptions by Country'
                )
                fig_regional_subscriptions.update_xaxes(tickangle=45)
                st.plotly_chart(fig_regional_subscriptions, use_container_width=True)
            
            with col2:
                fig_regional_revenue = px.bar(
                    regional_performance.head(10),
                    x='Country',
                    y='Revenue',
                    title='Revenue by Country'
                )
                fig_regional_revenue.update_xaxes(tickangle=45)
                st.plotly_chart(fig_regional_revenue, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Dashboard Features:**")
st.sidebar.markdown("• 📊 Real-time analytics")
st.sidebar.markdown("• 👥 Customer insights")
st.sidebar.markdown("• 🍽️ Program performance")
st.sidebar.markdown("• 📈 Subscription tracking")
st.sidebar.markdown("• 🍴 Meal analysis")
st.sidebar.markdown("• 🌍 Geographic insights")

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Source:**")
st.sidebar.markdown("Diet Program Management System")

# Add download functionality
if st.sidebar.button("📥 Download Sample Data"):
    # Create sample data for download
    sample_data = {}
    
    if customers_df is not None and not customers_df.empty:
        sample_data['customers'] = customers_df.head(1000)
    
    if diet_programs_df is not None and not diet_programs_df.empty:
        sample_data['diet_programs'] = diet_programs_df
    
    if customers_subscribers_df is not None and not customers_subscribers_df.empty:
        sample_data['customers_subscribers'] = customers_subscribers_df.head(1000)
    
    if selected_items_df is not None and not selected_items_df.empty:
        sample_data['selected_items'] = selected_items_df.head(1000)
    
    # Create Excel file with multiple sheets
    with pd.ExcelWriter('diet_program_analytics.xlsx', engine='openpyxl') as writer:
        for name, data in sample_data.items():
            data.to_excel(writer, sheet_name=name, index=False)
    
    st.sidebar.success("Data downloaded as 'diet_program_analytics.xlsx'")

# Data info
st.sidebar.markdown("---")
st.sidebar.markdown("**Data Info:**")
if customers_df is not None and not customers_df.empty:
    st.sidebar.markdown(f"• Customers: {len(customers_df):,}")
if diet_programs_df is not None and not diet_programs_df.empty:
    st.sidebar.markdown(f"• Programs: {len(diet_programs_df):,}")
if customers_subscribers_df is not None and not customers_subscribers_df.empty:
    st.sidebar.markdown(f"• Subscriptions: {len(customers_subscribers_df):,}")
if selected_items_df is not None and not selected_items_df.empty:
    st.sidebar.markdown(f"• Selected Items: {len(selected_items_df):,}") 