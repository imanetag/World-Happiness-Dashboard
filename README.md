# World Happiness Dashboard

## Project Description

The **World Happiness Dashboard** is an interactive and comprehensive Streamlit-based application designed for shareholders to explore the World Happiness dataset. It provides deep insights into global happiness trends, key drivers, and regional comparisons. The dashboard enables users to interact with dynamic visualizations to support data-driven decision-making.

This dashboard features advanced visualizations and analysis across multiple tabs, offering users the ability to:

- Analyze global happiness trends over time.
- Explore regional disparities and their key contributing factors.
- Compare countries and regions based on happiness and other related metrics.

## Features

### Tab 1: Overview

- **Summary Statistics**:
  - Displays key statistics, including the number of countries, average happiness score, and year range.
- **Top and Bottom 10 Countries**:
  - Interactive visualizations of countries with the highest and lowest happiness scores.

### Tab 2: Global Trends

- **Happiness Score Trends**:
  - Interactive line plot showing happiness trends over time.
- **Dynamic Filtering**:
  - Users can filter by region or country to customize the analysis.
- **Choropleth Map**:
  - A geographical visualization of happiness scores by country for the selected year.

### Tab 3: Regional Analysis

- **Key Drivers of Happiness**:
  - Bubble plot to visualize the relationship between happiness scores and economic/social factors.
- **Correlation Analysis**:
  - Scatter matrix plot and heatmap to explore correlations between variables.
- **Univariate Analysis**:
  - Histograms for key variables and time-series trends for happiness scores by region.

### Tab 4: Country Comparison

- **Regional Factors**:
  - Bar plots displaying factors contributing to happiness by region.
- **Trend Analysis**:
  - Line plots to analyze trends of selected factors for multiple regions.
- **Distribution Analysis**:
  - Box plots and histograms showing happiness score distributions.
- **Radar Charts**:
  - Comparative radar charts for selected countries and regions.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/imanetag/world-happiness-dashboard.git
   ```

2. Navigate to the project directory:

   ```bash
   cd world-happiness-dashboard
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:

   ```bash
   streamlit run Dashboard_Report_Happiness.py 
   ```

## Dataset

The dashboard uses the **World Happiness Report** dataset. Ensure the dataset is placed in the appropriate directory before running the application. The dataset contains the following key variables:

- **Happiness Score**
- **Social Support**
- **Life Expectancy**
- **Generosity**
- **Economy GDP**
- **Government Corruption**
- **Freedom**

### Dataset Source
The dataset used is sourced from the [World Happiness Report](https://worldhappiness.report/). For this project, the 2023 version of the dataset has been utilized, which provides annual happiness scores and associated variables for countries worldwide.

## Usage

1. Launch the dashboard using Streamlit.
2. Navigate through the tabs to explore different analyses.
3. Use filters and selections to customize the visualizations.
4. Export or download visualizations for reporting purposes.

## Dependencies

- Python 3.8+
- Streamlit
- Plotly
- Pandas
- NumPy
- Geopandas (for geographical analysis)

Refer to `requirements.txt` for the complete list of dependencies.

## Visual Preview

Below is a sample screenshot of the dashboard:

![Dashboard Preview](preview.png)


## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature description"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

