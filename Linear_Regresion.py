import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* 🌸 Soft Angular Gradient Background */
    .stApp {
        background: conic-gradient(
            from 180deg at 50% 50%, 
            #f9f9f9, 
            #fdf6f0, 
            #f0f9f9, 
            #f6f0fd, 
            #f9f9f9
        );
        min-height: 100vh;
    }
    
    .main-header {
        font-size: 3rem;
        color: #333;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: rgba(255,255,255,0.95);
    }
    
    .stButton>button {
        background-color: #6ca0dc;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 400;
        transition: background 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #ffb347;
        color: white;
    }
    
    .marquee {
        height: 50px;
        overflow: hidden;
        position: relative;
        background: linear-gradient(90deg, #a8dadc, #ffb347);
        color: white;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    .marquee p {
        position: absolute;
        width: 100%;
        height: 100%;
        margin: 0;
        line-height: 50px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        transform: translateX(100%);
        animation: marquee 15s linear infinite;
    }
    
    @keyframes marquee {
        0%   { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
    
    .interpretation-box {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #6ca0dc;
        margin-top: 1rem;
    }
    
    .step-header {
        color: #333;
        border-bottom: 2px solid #ffb347;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# Marquee header
st.markdown("""
<div class="marquee">
    <p>📊 Linear Regression Analysis Tool | Upload Your Data & Get Instant Results 📈</p>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">Linear Regression Calculator</h1>', unsafe_allow_html=True)

def main():
    # File upload
    uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx'], help="Your Excel file should have X values in the first column and Y values in the second column")
    
    if uploaded_file is not None:
        try:
            # Read the Excel file
            df = pd.read_excel(uploaded_file)
            st.markdown('<h2 class="step-header">Uploaded Data</h2>', unsafe_allow_html=True)
            st.dataframe(df.style.format("{:.4f}").set_properties(**{'background-color': '#f0f2f6', 'color': 'black', 'border': '1px solid white'}))
            
            # Check if we have the required columns
            if len(df.columns) < 2:
                st.error("Excel file must have at least two columns (X and Y values)")
                return
                
            # Use first two columns as X and Y
            x_col, y_col = df.columns[0], df.columns[1]
            x = df[x_col].astype(float).values
            y = df[y_col].astype(float).values
            
            if len(x) != len(y):
                st.error("X and Y must have the same number of data points")
                return
                
            n = len(x)
            
            # Create calculation table
            st.markdown('<h2 class="step-header">Step 1: Calculate Required Values</h2>', unsafe_allow_html=True)
            calc_df = pd.DataFrame({
                'X': x,
                'Y': y,
                'X²': x**2,
                'Y²': y**2,
                'X·Y': x*y
            })
            
            # Add sum row
            sum_row = {
                'X': calc_df['X'].sum(),
                'Y': calc_df['Y'].sum(),
                'X²': calc_df['X²'].sum(),
                'Y²': calc_df['Y²'].sum(),
                'X·Y': calc_df['X·Y'].sum()
            }
            
            # Display calculation table with styling
            st.dataframe(calc_df.style.format("{:.4f}").set_table_styles(
                [{'selector': 'th', 'props': [('background-color', '#1f77b4'), ('color', 'white')]}]
            ))
            
            # Display sum row
            st.markdown('<h2 class="step-header">Step 2: Calculate Sums (Σ)</h2>', unsafe_allow_html=True)
            sum_df = pd.DataFrame([sum_row], index=['Σ'])
            st.dataframe(sum_df.style.format("{:.4f}").set_table_styles(
                [{'selector': 'th', 'props': [('background-color', '#ff7f0e'), ('color', 'white')]}]
            ))
            
            # Extract sum values
            Σx = sum_row['X']
            Σy = sum_row['Y']
            Σx2 = sum_row['X²']
            Σy2 = sum_row['Y²']
            Σxy = sum_row['X·Y']
            
            # Show formulas
            st.markdown('<h2 class="step-header">Step 3: Linear Regression Formulas</h2>', unsafe_allow_html=True)
            st.latex(r"a = \frac{(\Sigma y)(\Sigma x^2) - (\Sigma x)(\Sigma xy)}{n(\Sigma x^2) - (\Sigma x)^2}")
            st.latex(r"b = \frac{n(\Sigma xy) - (\Sigma x)(\Sigma y)}{n(\Sigma x^2) - (\Sigma x)^2}")
            
            # Calculate denominator (common for both a and b)
            denominator = n * Σx2 - Σx**2
            
            if denominator == 0:
                st.error("Denominator is zero. Cannot calculate regression coefficients.")
                return
                
            # Calculate a (intercept)
            numerator_a = (Σy * Σx2) - (Σx * Σxy)
            a = numerator_a / denominator
            
            # Calculate b (slope)
            numerator_b = n * Σxy - Σx * Σy
            b = numerator_b / denominator
            
            # Show calculations in formula format
            st.markdown('<h2 class="step-header">Step 4: Calculate Coefficients</h2>', unsafe_allow_html=True)
            
            # Common denominator
            st.latex(f"n(\\Sigma x^2) - (\\Sigma x)^2 = {n}({Σx2}) - ({Σx})^2 = {denominator}")
            
            # Intercept (a)
            st.latex(f"a = \\frac{{(\\Sigma y)(\\Sigma x^2) - (\\Sigma x)(\\Sigma xy)}}{{n(\\Sigma x^2) - (\\Sigma x)^2}} = \\frac{{{Σy} \\times {Σx2} - {Σx} \\times {Σxy}}}{{{denominator}}} = \\frac{{{numerator_a}}}{{{denominator}}} = {a:.6f}")
            
            # Slope (b)
            st.latex(f"b = \\frac{{n(\\Sigma xy) - (\\Sigma x)(\\Sigma y)}}{{n(\\Sigma x^2) - (\\Sigma x)^2}} = \\frac{{{n} \\times {Σxy} - {Σx} \\times {Σy}}}{{{denominator}}} = \\frac{{{numerator_b}}}{{{denominator}}} = {b:.6f}")
            
            # Final equation
            st.markdown('<h2 class="step-header">Step 5: Final Linear Regression Equation</h2>', unsafe_allow_html=True)
            st.latex(f"y = {a:.6f} + {b:.6f}x")
            
            # Calculate predicted values
            y_pred = a + b * x
            
            # Calculate R-squared components
            y_mean = np.mean(y)
            ss_res = np.sum((y - y_pred)**2)  # Sum of squared residuals
            ss_tot = np.sum((y - y_mean)**2)  # Total sum of squares
            
            # Create table for R² calculation
            st.markdown('<h2 class="step-header">Step 6: Calculate R² (Goodness of Fit)</h2>', unsafe_allow_html=True)
            
            r2_df = pd.DataFrame({
                'Y': y,
                'Ŷ (Predicted)': y_pred,
                'Y - Ŷ': y - y_pred,
                '(Y - Ŷ)²': (y - y_pred)**2,
                'Y - ȳ': y - y_mean,
                '(Y - ȳ)²': (y - y_mean)**2
            })
            
            st.dataframe(r2_df.style.format("{:.4f}").set_table_styles(
                [{'selector': 'th', 'props': [('background-color', '#2ca02c'), ('color', 'white')]}]
            ))
            
            # Add sum row for R² calculation
            r2_sum_row = {
                'Y': r2_df['Y'].sum(),
                'Ŷ (Predicted)': r2_df['Ŷ (Predicted)'].sum(),
                'Y - Ŷ': r2_df['Y - Ŷ'].sum(),
                '(Y - Ŷ)²': r2_df['(Y - Ŷ)²'].sum(),
                'Y - ȳ': r2_df['Y - ȳ'].sum(),
                '(Y - ȳ)²': r2_df['(Y - ȳ)²'].sum()
            }
            
            st.write("Sums:")
            r2_sum_df = pd.DataFrame([r2_sum_row], index=['Σ'])
            st.dataframe(r2_sum_df.style.format("{:.4f}").set_table_styles(
                [{'selector': 'th', 'props': [('background-color', '#d62728'), ('color', 'white')]}]
            ))
            
            # Calculate R-squared
            r_squared = 1 - (ss_res / ss_tot)
            
            # Show R² calculation
            st.latex(f"SS_{{res}} = \\Sigma(y - \\hat{{y}})^2 = {ss_res:.6f}")
            st.latex(f"SS_{{tot}} = \\Sigma(y - \\bar{{y}})^2 = {ss_tot:.6f}")
            st.latex(f"R^2 = 1 - \\frac{{SS_{{res}}}}{{SS_{{tot}}}} = 1 - \\frac{{{ss_res:.6f}}}{{{ss_tot:.6f}}} = {r_squared:.6f}")
            
            # Create the interactive regression plot with Plotly
            st.markdown('<h2 class="step-header">Step 7: Interactive Regression Visualization</h2>', unsafe_allow_html=True)
            
            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": False}]])
            
            # Add actual data points
            fig.add_trace(go.Scatter(
                x=x, y=y, 
                mode='markers', 
                name='Actual Data',
                marker=dict(color='#1f77b4', size=10, opacity=0.7),
                hovertemplate='<b>X</b>: %{x:.4f}<br><b>Y</b>: %{y:.4f}<extra></extra>'
            ))
            
            # Add regression line
            x_line = np.linspace(min(x), max(x), 100)
            y_line = a + b * x_line
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line, 
                mode='lines', 
                name=f'Regression Line: y = {a:.4f} + {b:.4f}x',
                line=dict(color='#ff7f0e', width=3),
                hovertemplate='<b>X</b>: %{x:.4f}<br><b>Predicted Y</b>: %{y:.4f}<extra></extra>'
            ))
            
            # Add mean line
            fig.add_trace(go.Scatter(
                x=[min(x), max(x)], 
                y=[y_mean, y_mean], 
                mode='lines', 
                name=f'Mean of Y: ȳ = {y_mean:.4f}',
                line=dict(color='#2ca02c', width=2, dash='dash'),
                hovertemplate='<b>Y Mean</b>: %{y:.4f}<extra></extra>'
            ))
            
            # Add predicted points
            fig.add_trace(go.Scatter(
                x=x, y=y_pred, 
                mode='markers', 
                name='Predicted Values',
                marker=dict(color='#d62728', symbol='x', size=10),
                hovertemplate='<b>X</b>: %{x:.4f}<br><b>Predicted Y</b>: %{y:.4f}<extra></extra>'
            ))
            
            # Add residual lines
            for i in range(len(x)):
                fig.add_trace(go.Scatter(
                    x=[x[i], x[i]], 
                    y=[y[i], y_pred[i]], 
                    mode='lines', 
                    name='Residuals' if i == 0 else "",
                    line=dict(color='gray', width=1, dash='dot'),
                    showlegend=(i == 0),
                    hovertemplate=f'<b>Residual</b>: {abs(y[i]-y_pred[i]):.4f}<extra></extra>'
                ))
            
                fig.update_layout(
                    title="Interactive Linear Regression Analysis",
                    xaxis_title=str(x_col),
                    yaxis_title=str(y_col),
                    hovermode='closest',
                    template='plotly_white',
                    height=600,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            st.markdown('<h2 class="step-header">Interpretation</h2>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="interpretation-box">
                <p><b>Slope (b = {b:.6f})</b>: For each unit increase in X, Y changes by {b:.6f} units.</p>
                <p><b>Intercept (a = {a:.6f})</b>: Represents the predicted value of Y when X is zero.</p>
                <p><b>R² value of {r_squared:.6f}</b>: Indicates that {r_squared*100:.2f}% of the variation in Y is explained by X.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Graph insights
            st.markdown('<h2 class="step-header">Graph Insights</h2>', unsafe_allow_html=True)
            st.markdown("""
            <div class="interpretation-box">
                <p><b>Hover over the graph</b> to see exact coordinate values and residuals.</p>
                <p><b>Blue points</b>: Your actual data points</p>
                <p><b>Orange line</b>: The calculated regression line that best fits your data</p>
                <p><b>Green dashed line</b>: The mean of all Y values</p>
                <p><b>Red X marks</b>: The predicted Y values based on the regression equation</p>
                <p><b>Gray dotted lines</b>: The residuals (vertical distances between actual and predicted values)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional insights based on R² value
            if r_squared > 0.7:
                st.success("The data points are closely clustered around the regression line, indicating a strong linear relationship.")
            elif r_squared > 0.5:
                st.warning("The data points show a moderate linear relationship with the regression line.")
            else:
                st.info("The data points are widely scattered around the regression line, indicating a weak linear relationship.")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
