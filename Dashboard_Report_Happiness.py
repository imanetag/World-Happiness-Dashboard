import subprocess
import sys

# List installed packages to verify if plotly is installed
subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True)
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import pandas as pd


# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('Data_Dashboard.csv')

data = load_data()

# Title and Description
st.title("🌏 World Happiness Dashboard")
st.markdown("""
This interactive dashboard will allow you to explore global happiness data.  
Analyze trends, identify key drivers, and gain insights into what makes countries happier.
""")

# Sidebar for filtering
st.sidebar.header("Filter Data")
years = data['Year'].unique()
selected_year = st.sidebar.selectbox("Select Year", years)
regions = data['Region'].unique()
selected_region = st.sidebar.multiselect("Select Regions", regions, default=regions)

# Filter dataset based on user input
filtered_data = data[(data['Year'] == selected_year) & (data['Region'].isin(selected_region))]

# Calculate Economic Stability Score
filtered_data['Economic Stability Score'] = filtered_data['Economy GPD'] - filtered_data['Government Corruption']

# Tabs for navigation
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Global and Regional Trends", "Global and Regional Analysis", "Country and Regional Comparison"])

# --- Tab 1: Overview ---
with tab1:
    st.header("Investment Opportunities and National Happiness")
    st.subheader("Key Metrics")

    avg_happiness = filtered_data['Happiness Score'].mean()
    avg_economic_stability = filtered_data['Economic Stability Score'].mean()
    num_countries = filtered_data['Country'].nunique()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Happiness Score", round(avg_happiness, 2))
    with col2:
        st.metric("Average Economic Stability Score", round(avg_economic_stability, 2))
    with col3:
        st.metric("Number of Countries", num_countries)

    # Display Top 10 Countries by Happiness Score
    st.subheader("Top 10 Countries by Happiness Score")
    top_10_countries = filtered_data[['Country', 'Happiness Score', 'Region', 'Economy GPD', 'Economic Stability Score']] \
        .sort_values(by="Happiness Score", ascending=False).head(10)
    st.table(top_10_countries)

    # Display Bottom 10 Countries by Happiness Score
    st.subheader("Bottom 10 Countries by Happiness Score")
    bottom_10_countries = filtered_data[['Country', 'Happiness Score', 'Region', 'Economy GPD', 'Economic Stability Score']] \
        .sort_values(by="Happiness Score", ascending=True).head(10)
    st.table(bottom_10_countries)

    # Display Map with Economic Stability Score
    st.subheader(f"Global Happiness Map ({selected_year})")

    map_fig = px.choropleth(
        filtered_data,
        locations="Country",
        locationmode="country names",
        color="Happiness Score",
        hover_name="Country",
        hover_data=["Region", "Happiness Score", "Economy GPD", "Economic Stability Score"],  # Include Economic Stability Score
        color_continuous_scale="viridis",
        title=f"Happiness Scores by Country in {selected_year}",
    )

    st.plotly_chart(map_fig)
    # Happiness Flow by Region and Country (Sankey Diagram)
    st.subheader("Sankey Diagram: Happiness Flow by Region and Country")
    sankey_fig = px.sunburst(
        filtered_data,
        path=['Region', 'Country'],
        values='Happiness Score',
        title="Happiness Flow by Region and Country"
    )
    sankey_fig.update_layout(
        width=800, 
        height=600, 
        margin=dict(t=40, b=40, l=40, r=40)
    )
    st.plotly_chart(sankey_fig)
    #  Treemap for Happiness Scores by Region and Country
    st.subheader("Treemap: Happiness Scores by Region and Country")

# Check if the necessary columns are present
    if 'Region' in filtered_data.columns and 'Country' in filtered_data.columns and 'Happiness Score' in filtered_data.columns:
    
    # Create the treemap
        treemap_fig = px.treemap(
            filtered_data,
            path=['Region', 'Country'],
            values='Happiness Score',
            title="Happiness Scores by Region and Country",
            color='Happiness Score',  # Color based on Happiness Score
            color_continuous_scale='Viridis',  # Choose a color scale
            hover_data=['Region', 'Country', 'Happiness Score'],  # Display detailed info on hover
        )
    
        # Update layout for better visualization
        treemap_fig.update_layout(
            template="plotly_white",  # Choose a theme (optional)
            margin=dict(t=50, b=50, l=50, r=50),  # Adjust margins for better spacing
            width=1000,  # Increase the width of the treemap
            height=800,  # Increase the height of the treemap
        )

        # Display the interactive treemap plot in Streamlit
        st.plotly_chart(treemap_fig)
    else:
        st.error("Required columns ('Region', 'Country', 'Happiness Score') are missing in the data.")


# --- Tab 2: Global Trends ---
with tab2:
    # Regional Insights
    st.subheader("Regional Breakdown")
    avg_happiness_region = filtered_data.groupby('Region')['Happiness Score'].mean().reset_index()

    region_fig = px.bar(
        avg_happiness_region,
        x="Region",
        y="Happiness Score",
        color="Region",
        title="Average Happiness Score by Region",
    )
    st.plotly_chart(region_fig)
    # Line Plot for Happiness by Year
    # 1. Group the data by Year to compute average Happiness Score
    yearly_avg = data.groupby('Year')['Happiness Score'].mean().reset_index()

    st.subheader("Happiness Score by Year")
    year_fig = px.line(
        yearly_avg,
        x='Year',
        y='Happiness Score',
        title="Happiness Scores by Year",
        labels={'Year': 'Year', 'Happiness Score': 'Average Happiness Score'},
    )
    
    # Update the layout to set the x-axis to display whole years
    year_fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            tick0=yearly_avg['Year'].min(),  # Start at the first year
            dtick=1  # Increment by 1 year
        )
    )
    st.plotly_chart(year_fig)

    # --- Stacked Bar Plot of Happiness Scores by Region and Year ---
    st.subheader("Happiness Scores by Region and Year")
    stacked_bar_figs = px.bar(
        data,
        x="Year",
        y="Happiness Score",
        color="Region",
        title="Happiness Scores by Region and Year",
        barmode='stack'
    )

    # Update the layout to set the x-axis to display whole years
    stacked_bar_figs.update_layout(
        xaxis=dict(
            tickmode='linear',
            tick0=data['Year'].min(),  # Start at the first year
            dtick=1  # Increment by 1 year
        )
    )

    st.plotly_chart(stacked_bar_figs)
    # Trend Analysis
    trend_countries = st.multiselect("Select Countries for Trend Analysis", data['Country'].unique())

    column_options = ['Happiness Score', 'Economy GPD', 'Life Expectancy', 'Social Support', 'Generosity', 'Freedom', 'Government Corruption']
    selected_column = st.selectbox("Select a Column for Trend Analysis", column_options)

    if trend_countries:
        trend_data = data[data['Country'].isin(trend_countries)]

        if selected_column:
            trend_data = trend_data[['Year', 'Country', selected_column]]
            trend_fig = px.line(
                trend_data,
                x="Year",
                y=selected_column,
                color="Country",
                title=f"Trend Analysis of {selected_column}"
            )
            st.plotly_chart(trend_fig)
        else:
            st.write("Please select a column for trend analysis.")
    # Interactive Density Plot for Happiness Scores by Region
    st.subheader("Density Plot for Happiness Scores by Region")


    # Prepare the data for each region
    region_data = [data[data['Region'] == region]['Happiness Score'] for region in data['Region'].unique()]

    # Create the density plot using Plotly's kde function
    fig = ff.create_distplot(
        region_data,  # Data for each region
        group_labels=data['Region'].unique(),  # Labels for each region
        show_hist=False,  # Turn off the histogram
        show_rug=False,   # Turn off the rug plot (small ticks)
        colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],  # Define specific colors for regions
    )

# Update the layout for the plot (axis titles, etc.)
    fig.update_layout(
        title="Density Plot for Happiness Scores by Region",
        xaxis_title="Happiness Score",
        yaxis_title="Density",
        template="plotly_dark",  # Optional: dark theme for the plot
        height=400,
        legend_title="Regions",  # Title for the legend
        xaxis=dict(
            tickmode='linear',  # Ensure the x-axis ticks are linear
            dtick=1,  # Set the step size for the x-axis ticks to 1 (you can adjust this as needed)
        )
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)
    # Parallel Coordinates Plot
    st.subheader("Parallel Coordinates Plot of Key Variables")
    key_variables = ['Happiness Score', 'Life Expectancy', 'Government Corruption', 'Social Support', 'Economy GPD']

    fig_parallel_coordinates = px.parallel_coordinates(
        data,
        dimensions=key_variables,
        color='Happiness Score',  # Color by Happiness Score
        color_continuous_scale=px.colors.sequential.Plasma,  # Use a single consistent color scale
        labels={  # Custom labels for axes
            'Happiness Score': 'Happiness Score',
            'Life Expectancy': 'Life Expectancy (Years)',
            'Government Corruption': 'Government Corruption',
            'Social Support': 'Social Support',
            'Economy GPD': 'Economy GDP (Billions)'
        },
    )

    fig_parallel_coordinates.update_traces(
        line_coloraxis='coloraxis',
    )

    fig_parallel_coordinates.update_layout(
        coloraxis=dict(
            colorscale='Plasma',  # Experiment with different color scales
            colorbar=dict(title='Happiness Score')
        )
    )

    st.plotly_chart(fig_parallel_coordinates, use_container_width=True)

# --- Tab 3: Regional Analysis ---
with tab3:
    st.header("Key Drivers of Happiness")

    # Bubble Plot for Happiness vs Economic Factors
    economic_factor = st.selectbox(
        "Select an Economic Factor", 
        ["Economy GPD", "Life Expectancy", "Generosity", "Government Corruption", "Freedom"]
    )

    bubble_fig = px.scatter(
        filtered_data,
        x=economic_factor,  # Use the selected economic factor as x-axis
        y="Happiness Score",
        size="Happiness Rank",  # Adjust bubble size based on the rank
        color="Region",  # Color the bubbles based on region
        hover_name="Country",  # Display country when hovering
        title=f"Happiness vs {economic_factor}",
        hover_data=["Happiness Score", "Happiness Rank", "Region", "Country"]  # Include additional columns in hover
    )

    st.plotly_chart(bubble_fig)

    # Scatter Matrix Plot
    st.header("Correlation Matrix")

    numeric_cols = ['Happiness Score', 'Social Support', 'Life Expectancy', 'Generosity', 
                    'Economy GPD', 'Government Corruption', 'Freedom']

    filtered_data = filtered_data.dropna(subset=numeric_cols)

    abbreviated_cols = {
        'Happiness Score': 'Hap. Score',
        'Social Support': 'Social Supp.',
        'Life Expectancy': 'Life Exp.',
        'Generosity': 'Generosity',
        'Economy GPD': 'Economy',
        'Government Corruption': 'Gov. Corr.',
        'Freedom': 'Freedom'
    }

    fig_scatter_matrix = px.scatter_matrix(
        filtered_data,
        dimensions=numeric_cols,
        color='Region',
        labels=abbreviated_cols
    )

    fig_scatter_matrix.update_layout(
        autosize=False,
        width=2500,  # Wide enough to spread horizontally
        height=650,  # Set a taller height for better vertical space
        margin=dict(t=50, b=100, l=50, r=50),  # Balanced margins
        plot_bgcolor="white",  # Set plot background color
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),  # Hover styling
        showlegend=True,  # Ensure the legend is visible
        legend=dict(title="Region", font=dict(size=12), orientation="h", x=0.5, xanchor="center")  # Centered horizontal legend
    )

    fig_scatter_matrix.update_xaxes(
        tickangle=45,  # Rotate x-axis labels for better readability
        tickfont=dict(size=10, color="black"),  # Font size and color
        title=dict(font=dict(size=12))  # Title font size
    )
    fig_scatter_matrix.update_yaxes(
        tickfont=dict(size=10, color="black"),  # Font size and color
        title=dict(font=dict(size=12))  # Title font size
    )

    fig_scatter_matrix.update_traces(marker=dict(size=5, opacity=0.8), diagonal_visible=True)

    st.plotly_chart(fig_scatter_matrix)
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")

    correlation_data = data[['Happiness Score', 'Economy GPD', 'Life Expectancy', 'Freedom', 'Generosity', 'Social Support']]
    corr_matrix = correlation_data.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Sunset',  # Elegant color scale
        colorbar=dict(
            title='Correlation',
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=['-1', '-0.5', '0', '0.5', '1'],
            tickangle=0,
            ticks='outside',
            tickfont=dict(size=12, family='Arial', color='rgb(90, 90, 90)'),
            title_font=dict(size=14, family='Arial', color='rgb(70, 70, 70)')
        ),
        showscale=True
    ))

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(corr_matrix.columns))),
            ticktext=corr_matrix.columns,
            tickangle=45,  # Slight angle for readability
            tickfont=dict(size=14, family='Arial', color='rgb(60, 60, 60)'),
            showgrid=False,  # No grid lines
            zeroline=False,  # No zero line
            showticklabels=True
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(corr_matrix.columns))),
            ticktext=corr_matrix.columns,
            tickangle=-45,  # Slight angle for readability
            tickfont=dict(size=14, family='Arial', color='rgb(60, 60, 60)'),
            showgrid=False,  # No grid lines
            zeroline=False,  # No zero line
            showticklabels=True
        ),
        plot_bgcolor='white',  # No background color for the plot
        paper_bgcolor='white',  # White background for the entire figure for a clean look
        margin=dict(l=50, r=50, b=50, t=80),  # Adequate spacing around the figure
    )

    st.plotly_chart(fig)
    # List of key variables for univariate analysis
    key_variables = [
        "Happiness Score",
        "Social Support",
        "Life Expectancy",
        "Generosity",
        "Economy GPD",
        "Government Corruption",
        "Freedom"
    ]

    # Define a color palette
    color_palette = [
        "skyblue", "lightcoral", "mediumseagreen", "gold", "mediumpurple", "tomato", "darkorange"
    ]
     
    st.subheader("Time Series: Happiness Score Trends by Region")
    time_series_data = data.groupby(['Year', 'Region'])['Happiness Score'].mean().reset_index()
    time_series_fig = px.line(
        time_series_data,
        x='Year', 
        y='Happiness Score', 
        color='Region', 
        title="Happiness Score Trends by Region", 
        labels={'Happiness Score': 'Average Happiness Score', 'Year': 'Year'},
        markers=True
    )
    st.plotly_chart(time_series_fig)
        # Univariate Analysis: Histograms for Key Variables
    st.subheader("Univariate Analysis Histograms ")
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=key_variables,
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    for i, (var, color) in enumerate(zip(key_variables, color_palette)):
        row = i // 3 + 1
        col = i % 3 + 1
        fig.add_trace(
            go.Histogram(
                x=data[var],
                name=var,
                nbinsx=20,
                marker=dict(color=color),
                opacity=0.7,
                histnorm="density"
            ),
            row=row, col=col
        )

    fig.update_layout(
        height=900,
        width=1100,
        title_text="Univariate Analysis Histogram",
        template="plotly_white",
        showlegend=False,
        font=dict(size=12)
    )
    fig.update_xaxes(title_text="Value")
    fig.update_yaxes(title_text="Density")
    st.plotly_chart(fig, key="grid_histograms")
# --- Tab 4: Country Comparison ---
with tab4:
    st.header("Factors Contributing to Happiness by Region")
    
    # Bar plot for the factors contributing to happiness by region
    fig_bar_factors = go.Figure()
    factors = ['Social Support', 'Life Expectancy', 'Generosity', 'Economy GPD', 'Government Corruption', 'Freedom']
    
    for factor in factors:
        fig_bar_factors.add_trace(go.Bar(x=filtered_data['Region'], y=filtered_data[factor], name=factor))

    fig_bar_factors.update_layout(
        barmode='group',
        xaxis_tickangle=-45,
        title="Factors Contributing to Happiness by Region"
    )
    
    st.plotly_chart(fig_bar_factors)

    # Trend Analysis for Specific Regions and Columns
    st.subheader("Trend Analysis for Selected Regions")
    
    regions = st.multiselect("Select Regions for Trend Analysis", filtered_data['Region'].unique())
    column_for_trend = st.selectbox("Select a Column for Trend Analysis", 
                                    ["Economy GPD", "Life Expectancy", "Generosity", 
                                     "Government Corruption", "Freedom", "Social Support"])

    trend_data = data[data['Region'].isin(regions)]

    if not trend_data.empty:
        fig_trend = go.Figure()

        for region in regions:
            region_data = trend_data[trend_data['Region'] == region]
            region_trend_data = region_data.groupby('Year')[column_for_trend].mean().reset_index()

            fig_trend.add_trace(go.Scatter(
                x=region_trend_data['Year'],
                y=region_trend_data[column_for_trend],
                mode='lines+markers',
                name=f"{region} - {column_for_trend} Trend",
            ))

        fig_trend.update_layout(
            title=f"{column_for_trend} Trend for Selected Regions",
            xaxis_title="Year",
            yaxis_title=column_for_trend,
            template="plotly_dark",
            showlegend=True,
            plot_bgcolor="rgba(0, 0, 0, 0)",
            margin=dict(t=50, b=100, l=50, r=50),
        )

        st.plotly_chart(fig_trend)
    else:
        st.write("No data available for the selected regions.")

    # Distribution of Happiness Scores
    st.subheader("Distribution of Happiness Scores")
    hist_fig = px.histogram(
        filtered_data,
        x="Happiness Score",
        nbins=30,
        color="Region",
        title="Distribution of Happiness Scores",
    )
    st.plotly_chart(hist_fig)

    # Boxplot for Happiness by Region
    st.subheader("Happiness Distribution by Region")
    box_fig = px.box(
        filtered_data,
        x="Region",
        y="Happiness Score",
        color="Region",
        title="Happiness Distribution by Region",
    )
    st.plotly_chart(box_fig)

    # Create two columns to display the radar charts side by side
    col1, col2 = st.columns(2)

    # --- Radar Chart for Country Comparison (Placed in the first column) ---
    with col1:
        st.subheader("Radar Chart for Country Comparison")

        countries = st.multiselect("Select Countries for Comparison", filtered_data['Country'].unique())

        if countries:
            fig_country = go.Figure()

            color_scale = px.colors.qualitative.Plotly
            max_value_country = 0

            for idx, country in enumerate(countries):
                country_data = filtered_data[filtered_data['Country'] == country].iloc[0]
                country_values = [country_data[factor] for factor in ['Social Support', 'Life Expectancy', 'Generosity', 
                                                                     'Economy GPD', 'Government Corruption', 'Freedom']]

                max_value_country = max(max_value_country, max(country_values))

                fig_country.add_trace(go.Scatterpolar(
                    r=country_values + [country_values[0]],  
                    theta=['Social Support', 'Life Expectancy', 'Generosity', 
                           'Economy GPD', 'Government Corruption', 'Freedom'] + ['Social Support'],  
                    fill='toself',  
                    name=country,
                    line=dict(color=color_scale[idx % len(color_scale)]),  
                ))

            fig_country.update_layout(
                title="Radar Chart for Selected Countries",
                width=600,
                height=600,
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max_value_country + 1]
                    )
                ),
                showlegend=True,
                template="plotly",
            )

            st.plotly_chart(fig_country)
        else:
            st.write("Please select at least one country to display the radar chart.")

    # --- Radar Chart for Region Comparison (Placed in the second column) ---
    with col2:
        st.subheader("Radar Chart for Region Comparison")

        regions = st.multiselect("Select Regions for Comparison", filtered_data['Region'].unique())

        if regions:
            # Filter data based on selected regions
            filtered_region_data = filtered_data[filtered_data['Region'].isin(regions)]

            # Calculate the average values for each factor per region
            region_avg_values = filtered_region_data.groupby('Region')[['Social Support', 'Life Expectancy', 'Generosity', 
                                                                       'Economy GPD', 'Government Corruption', 'Freedom']].mean().reset_index()

            # Create a radar chart for each region
            fig_region = go.Figure()

            # Define a color scale for regions
            color_scale = px.colors.qualitative.Plotly  # Automatically assigns colors from Plotly color palette

            # Store the maximum value for axis range
            max_value_region = 0

            for idx, region in enumerate(regions):
                region_values = region_avg_values[region_avg_values['Region'] == region][['Social Support', 'Life Expectancy', 
                                                                                      'Generosity', 'Economy GPD', 
                                                                                      'Government Corruption', 'Freedom']].values.flatten()

                max_value_region = max(max_value_region, max(region_values))  # Track the maximum value

                # Add trace for the region's radar chart with filling
                fig_region.add_trace(go.Scatterpolar(
                    r=region_values.tolist() + [region_values[0]],  # Close the loop for the radar chart
                    theta=['Social Support', 'Life Expectancy', 'Generosity', 
                           'Economy GPD', 'Government Corruption', 'Freedom'] + ['Social Support'],  # Close the angle as well
                    fill='toself',  # Fill the area inside the radar chart
                    name=region,
                    line=dict(color=color_scale[idx % len(color_scale)])  # Color is automatically assigned based on region index
                ))

            # Update layout with annotations and formatting
            fig_region.update_layout(
                title="Radar Chart for Selected Regions",
                width=600,  # Set the width of the chart
                height=600,  # Set the height of the chart
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max_value_region + 1]  # Dynamically adjust the range based on the max value for the selected regions
                    )
                ),
                showlegend=True,
                template="plotly",
                annotations=[
                    dict(
                        x=0.5, y=1.1,  # Adjust position for "Factors (θ)"
                        text="Factors (θ)",  # Theta axis represents the factors
                        showarrow=False,
                        font=dict(size=14, color='white'),
                        align="center"
                    ),
                    dict(
                        x=0.5, y=-0.1,  # Adjust position for "Scores (r)"
                        text="Scores (r)",  # Radial axis represents the scores
                        showarrow=False,
                        font=dict(size=14, color='white'),
                        align="center"
                    )
                ]
            )

            st.plotly_chart(fig_region)
        else:
            st.write("Please select at least one region to display the radar chart.")
