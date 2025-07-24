import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class EnhancedDataVisualizer:
    """Enhanced data visualization with intelligent chart recommendations and interactive features"""
    
    def __init__(self):
        self.color_palettes = {
            'default': px.colors.qualitative.Set3,
            'business': px.colors.qualitative.Pastel1,
            'dark': px.colors.qualitative.Dark24,
            'bright': px.colors.qualitative.Vivid
        }
        self.current_palette = self.color_palettes['default']
    
    def create_visualization(self, results: List[Dict[str, Any]], 
                           viz_suggestion: Dict[str, Any],
                           natural_query: str) -> None:
        """Create enhanced visualization with interactive controls"""
        
        if not results:
            st.info("📊 No data available for visualization.")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        viz_type = viz_suggestion.get('type', 'table')
        
        # Display recommendation info
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"📊 **{viz_suggestion.get('title', viz_type.title())}**: {viz_suggestion.get('description', 'Visualization of your data')}")
            with col2:
                color_theme = st.selectbox(
                    "Color Theme", 
                    options=['default', 'business', 'dark', 'bright'],
                    key=f"color_theme_{viz_type}"
                )
                self.current_palette = self.color_palettes[color_theme]
        
        try:
            if viz_type == 'bar':
                self._create_enhanced_bar_chart(df, natural_query, viz_suggestion)
            elif viz_type == 'line':
                self._create_enhanced_line_chart(df, natural_query, viz_suggestion)
            elif viz_type == 'scatter':
                self._create_enhanced_scatter_plot(df, natural_query, viz_suggestion)
            elif viz_type == 'pie':
                self._create_enhanced_pie_chart(df, natural_query, viz_suggestion)
            elif viz_type == 'histogram':
                self._create_enhanced_histogram(df, natural_query, viz_suggestion)
            else:
                self._create_enhanced_table(df, natural_query, viz_suggestion)
                
        except Exception as e:
            logger.error(f"Error creating {viz_type} visualization: {str(e)}")
            st.error(f"❌ Could not create {viz_type} chart: {str(e)}")
            st.info("💡 Showing data as table instead.")
            self._create_enhanced_table(df, natural_query, viz_suggestion)
    
    def _create_enhanced_bar_chart(self, df: pd.DataFrame, query: str, suggestion: Dict) -> None:
        """Create enhanced bar chart with interactive features"""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        if not numeric_cols or not text_cols:
            st.warning("📊 Bar chart requires both categorical and numeric data.")
            self._create_enhanced_table(df, query, suggestion)
            return
        
        # Interactive column selection
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("Category (X-axis)", text_cols, key="bar_x")
        with col2:
            y_col = st.selectbox("Value (Y-axis)", numeric_cols, key="bar_y")
        with col3:
            sort_by = st.selectbox("Sort by", ["Value (Desc)", "Value (Asc)", "Category", "None"], key="bar_sort")
        
        # Process data
        df_chart = df.copy()
        
        # Handle duplicates by aggregating
        if df_chart[x_col].duplicated().any():
            df_chart = df_chart.groupby(x_col)[y_col].sum().reset_index()
            st.info(f"🔄 Aggregated duplicate categories by summing {y_col}")
        
        # Apply sorting
        if sort_by == "Value (Desc)":
            df_chart = df_chart.nlargest(20, y_col)
        elif sort_by == "Value (Asc)":
            df_chart = df_chart.nsmallest(20, y_col)
        elif sort_by == "Category":
            df_chart = df_chart.sort_values(x_col).head(20)
        else:
            df_chart = df_chart.head(20)
        
        if len(df_chart) < len(df):
            st.caption(f"📊 Showing top {len(df_chart)} of {len(df)} categories")
        
        # Create interactive bar chart
        fig = px.bar(
            df_chart,
            x=x_col,
            y=y_col,
            title=f"{y_col} by {x_col}",
            color=y_col,
            color_continuous_scale=px.colors.sequential.Viridis,
            hover_data=[x_col, y_col]
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            showlegend=False,
            hovermode='x unified'
        )
        
        fig.update_traces(
            hovertemplate=f"<b>{x_col}</b>: %{{x}}<br><b>{y_col}</b>: %{{y:,.0f}}<extra></extra>"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show insights
        self._show_bar_chart_insights(df_chart, x_col, y_col)
    
    def _create_enhanced_line_chart(self, df: pd.DataFrame, query: str, suggestion: Dict) -> None:
        """Create enhanced line chart with trend analysis"""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        all_cols = df.columns.tolist()
        
        # Try to identify time/date columns
        date_cols = []
        for col in all_cols:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month', 'day']):
                date_cols.append(col)
        
        if not date_cols:
            date_cols = [all_cols[0]]  # Use first column as x-axis
        
        if not numeric_cols:
            st.warning("📈 Line chart requires numeric data.")
            self._create_enhanced_table(df, query, suggestion)
            return
        
        # Interactive controls
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("Time/Category (X-axis)", date_cols + all_cols, key="line_x")
        with col2:
            y_col = st.selectbox("Value (Y-axis)", numeric_cols, key="line_y")
        with col3:
            show_trend = st.checkbox("Show Trend Line", value=True, key="line_trend")
        
        # Process data
        df_chart = df.copy()
        
        # Sort by x column
        try:
            df_chart = df_chart.sort_values(x_col)
        except:
            pass
        
        # Create line chart
        fig = px.line(
            df_chart,
            x=x_col,
            y=y_col,
            title=f"{y_col} over {x_col}",
            markers=True,
            color_discrete_sequence=self.current_palette
        )
        
        # Add trend line if requested
        if show_trend and len(df_chart) > 2:
            try:
                # Calculate trend line
                x_numeric = pd.to_numeric(df_chart[x_col], errors='coerce')
                if not x_numeric.isna().all():
                    z = np.polyfit(x_numeric.dropna().index, df_chart[y_col].iloc[x_numeric.dropna().index], 1)
                    p = np.poly1d(z)
                    
                    fig.add_trace(go.Scatter(
                        x=df_chart[x_col],
                        y=p(x_numeric.fillna(method='linear')),
                        mode='lines',
                        name='Trend',
                        line=dict(dash='dash', color='red')
                    ))
            except:
                pass
        
        fig.update_layout(
            height=500,
            hovermode='x unified'
        )
        
        fig.update_traces(
            hovertemplate=f"<b>{x_col}</b>: %{{x}}<br><b>{y_col}</b>: %{{y:,.2f}}<extra></extra>"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show trend analysis
        self._show_line_chart_insights(df_chart, x_col, y_col)
    
    def _create_enhanced_scatter_plot(self, df: pd.DataFrame, query: str, suggestion: Dict) -> None:
        """Create enhanced scatter plot with correlation analysis"""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("🔍 Scatter plot requires at least 2 numeric columns.")
            self._create_enhanced_table(df, query, suggestion)
            return
        
        # Interactive controls
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
        with col2:
            y_col = st.selectbox("Y-axis", [col for col in numeric_cols if col != x_col], key="scatter_y")
        with col3:
            size_col = st.selectbox("Size by", ["None"] + numeric_cols, key="scatter_size")
        with col4:
            color_col = st.selectbox("Color by", ["None"] + text_cols + numeric_cols, key="scatter_color")
        
        # Prepare parameters
        size_param = None if size_col == "None" else size_col
        color_param = None if color_col == "None" else color_col
        
        # Limit color categories for readability
        if color_param and color_param in text_cols:
            unique_colors = df[color_param].nunique()
            if unique_colors > 15:
                st.warning(f"⚠️ Too many color categories ({unique_colors}). Using top 15.")
                top_categories = df[color_param].value_counts().head(15).index
                df_chart = df[df[color_param].isin(top_categories)].copy()
            else:
                df_chart = df.copy()
        else:
            df_chart = df.copy()
        
        # Create scatter plot
        fig = px.scatter(
            df_chart,
            x=x_col,
            y=y_col,
            size=size_param,
            color=color_param,
            title=f"{y_col} vs {x_col}",
            hover_data=df_chart.columns.tolist(),
            color_discrete_sequence=self.current_palette
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show correlation analysis
        self._show_scatter_insights(df_chart, x_col, y_col)
    
    def _create_enhanced_pie_chart(self, df: pd.DataFrame, query: str, suggestion: Dict) -> None:
        """Create enhanced pie chart with percentage breakdown"""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        if not numeric_cols or not text_cols:
            st.warning("🥧 Pie chart requires both categorical and numeric data.")
            self._create_enhanced_table(df, query, suggestion)
            return
        
        # Interactive controls
        col1, col2, col3 = st.columns(3)
        with col1:
            labels_col = st.selectbox("Categories", text_cols, key="pie_labels")
        with col2:
            values_col = st.selectbox("Values", numeric_cols, key="pie_values")
        with col3:
            max_slices = st.slider("Max slices", 3, 15, 8, key="pie_max")
        
        # Process data
        df_chart = df.copy()
        
        # Aggregate if needed
        if df_chart[labels_col].duplicated().any():
            df_chart = df_chart.groupby(labels_col)[values_col].sum().reset_index()
            st.info(f"🔄 Aggregated duplicate categories by summing {values_col}")
        
        # Limit to top categories
        if len(df_chart) > max_slices:
            df_top = df_chart.nlargest(max_slices - 1, values_col)
            others_sum = df_chart[~df_chart.index.isin(df_top.index)][values_col].sum()
            
            if others_sum > 0:
                others_row = pd.DataFrame({labels_col: ['Others'], values_col: [others_sum]})
                df_chart = pd.concat([df_top, others_row], ignore_index=True)
            else:
                df_chart = df_top
        
        # Create pie chart
        fig = px.pie(
            df_chart,
            values=values_col,
            names=labels_col,
            title=f"Distribution of {values_col} by {labels_col}",
            color_discrete_sequence=self.current_palette
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate="<b>%{label}</b><br>Value: %{value:,.0f}<br>Percentage: %{percent}<extra></extra>"
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show pie chart insights
        self._show_pie_chart_insights(df_chart, labels_col, values_col)
    
    def _create_enhanced_histogram(self, df: pd.DataFrame, query: str, suggestion: Dict) -> None:
        """Create enhanced histogram with distribution analysis"""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            st.warning("📊 Histogram requires numeric data.")
            self._create_enhanced_table(df, query, suggestion)
            return
        
        # Interactive controls
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("Column to analyze", numeric_cols, key="hist_x")
        with col2:
            bins = st.slider("Number of bins", 10, 50, 20, key="hist_bins")
        with col3:
            show_stats = st.checkbox("Show statistics", True, key="hist_stats")
        
        # Create histogram
        fig = px.histogram(
            df,
            x=x_col,
            nbins=bins,
            title=f"Distribution of {x_col}",
            color_discrete_sequence=self.current_palette
        )
        
        # Add mean line
        mean_val = df[x_col].mean()
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_val:.2f}"
        )
        
        fig.update_layout(
            height=500,
            bargap=0.1
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show distribution insights
        if show_stats:
            self._show_histogram_insights(df, x_col)
    
    def _create_enhanced_table(self, df: pd.DataFrame, query: str, suggestion: Dict) -> None:
        """Create enhanced table with sorting and filtering"""
        st.subheader("📋 Data Table")
        
        # Table controls
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Rows", f"{len(df):,}")
        with col2:
            st.metric("📋 Columns", len(df.columns))
        with col3:
            memory_usage = df.memory_usage(deep=True).sum() / 1024
            st.metric("💾 Memory", f"{memory_usage:.1f} KB")
        with col4:
            show_dtypes = st.checkbox("Show data types", key="table_dtypes")
        
        # Search functionality
        search_term = st.text_input("🔍 Search in data", key="table_search")
        if search_term:
            # Search across all string columns
            mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
            df_display = df[mask]
            st.caption(f"Found {len(df_display)} rows matching '{search_term}'")
        else:
            df_display = df
        
        # Column selector
        selected_columns = st.multiselect(
            "Select columns to display",
            options=df.columns.tolist(),
            default=df.columns.tolist()[:10],  # Show first 10 columns by default
            key="table_columns"
        )
        
        if selected_columns:
            df_display = df_display[selected_columns]
        
        # Display table
        if len(df_display) > 1000:
            st.warning(f"⚠️ Large dataset ({len(df_display):,} rows). Showing first 1000 rows.")
            st.dataframe(df_display.head(1000), use_container_width=True, height=400)
        else:
            st.dataframe(df_display, use_container_width=True, height=400)
        
        # Data types info
        if show_dtypes:
            with st.expander("📊 Column Information"):
                col_info = []
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    non_null = df[col].count()
                    null_count = len(df) - non_null
                    unique_count = df[col].nunique()
                    
                    # Sample values for categorical columns
                    if dtype == 'object' and unique_count <= 10:
                        sample_values = ', '.join(df[col].dropna().unique().astype(str)[:5])
                    else:
                        sample_values = "—"
                    
                    col_info.append({
                        'Column': col,
                        'Data Type': dtype,
                        'Non-Null': f"{non_null:,}",
                        'Null': f"{null_count:,}",
                        'Unique': f"{unique_count:,}",
                        'Sample Values': sample_values
                    })
                
                st.dataframe(pd.DataFrame(col_info), use_container_width=True, hide_index=True)
        
        # Summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            with st.expander("📈 Summary Statistics"):
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    def _show_bar_chart_insights(self, df: pd.DataFrame, x_col: str, y_col: str) -> None:
        """Show insights for bar charts"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total = df[y_col].sum()
            st.metric("📊 Total", f"{total:,.0f}")
        with col2:
            avg = df[y_col].mean()
            st.metric("📈 Average", f"{avg:,.1f}")
        with col3:
            top_category = df.loc[df[y_col].idxmax(), x_col]
            top_value = df[y_col].max()
            st.metric("🏆 Highest", f"{top_category}", f"{top_value:,.0f}")
        with col4:
            categories = len(df)
            st.metric("📋 Categories", categories)
    
    def _show_line_chart_insights(self, df: pd.DataFrame, x_col: str, y_col: str) -> None:
        """Show insights for line charts"""
        if len(df) > 1:
            first_val = df[y_col].iloc[0]
            last_val = df[y_col].iloc[-1]
            change = last_val - first_val
            pct_change = (change / first_val * 100) if first_val != 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📈 Start Value", f"{first_val:,.1f}")
            with col2:
                st.metric("📊 End Value", f"{last_val:,.1f}")
            with col3:
                st.metric("📉 Change", f"{change:,.1f}", f"{pct_change:+.1f}%")
            with col4:
                volatility = df[y_col].std()
                st.metric("📊 Volatility", f"{volatility:,.2f}")
    
    def _show_scatter_insights(self, df: pd.DataFrame, x_col: str, y_col: str) -> None:
        """Show insights for scatter plots"""
        if len(df) > 1:
            try:
                correlation = df[x_col].corr(df[y_col])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🔗 Correlation", f"{correlation:.3f}")
                with col2:
                    st.metric("📊 Data Points", f"{len(df):,}")
                with col3:
                    x_range = df[x_col].max() - df[x_col].min()
                    st.metric(f"📏 {x_col} Range", f"{x_range:,.1f}")
                with col4:
                    y_range = df[y_col].max() - df[y_col].min()
                    st.metric(f"📏 {y_col} Range", f"{y_range:,.1f}")
                
                # Correlation interpretation
                if abs(correlation) > 0.7:
                    correlation_desc = "Strong"
                elif abs(correlation) > 0.3:
                    correlation_desc = "Moderate"
                else:
                    correlation_desc = "Weak"
                
                direction = "positive" if correlation > 0 else "negative"
                st.info(f"📊 **Correlation Analysis**: {correlation_desc} {direction} correlation between {x_col} and {y_col}")
                
            except:
                st.info("📊 Unable to calculate correlation for this data")
    
    def _show_pie_chart_insights(self, df: pd.DataFrame, labels_col: str, values_col: str) -> None:
        """Show insights for pie charts"""
        total = df[values_col].sum()
        largest_slice = df.loc[df[values_col].idxmax()]
        smallest_slice = df.loc[df[values_col].idxmin()]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total", f"{total:,.0f}")
        with col2:
            st.metric("🍰 Slices", len(df))
        with col3:
            largest_pct = (largest_slice[values_col] / total) * 100
            st.metric("🏆 Largest", f"{largest_slice[labels_col]}", f"{largest_pct:.1f}%")
        with col4:
            smallest_pct = (smallest_slice[values_col] / total) * 100
            st.metric("📉 Smallest", f"{smallest_slice[labels_col]}", f"{smallest_pct:.1f}%")
    
    def _show_histogram_insights(self, df: pd.DataFrame, x_col: str) -> None:
        """Show insights for histograms"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            mean_val = df[x_col].mean()
            st.metric("📊 Mean", f"{mean_val:.2f}")
        with col2:
            median_val = df[x_col].median()
            st.metric("📈 Median", f"{median_val:.2f}")
        with col3:
            std_val = df[x_col].std()
            st.metric("📊 Std Dev", f"{std_val:.2f}")
        with col4:
            skewness = df[x_col].skew()
            st.metric("📊 Skewness", f"{skewness:.2f}")
        
        # Distribution interpretation
        if abs(skewness) < 0.5:
            dist_desc = "approximately normal"
        elif skewness > 0.5:
            dist_desc = "right-skewed (tail extends right)"
        else:
            dist_desc = "left-skewed (tail extends left)"
        
        st.info(f"📊 **Distribution Analysis**: The data appears to be {dist_desc}")
    
    def create_basic_visualization(self, df: pd.DataFrame, viz_type: str) -> None:
        """Create basic visualization without suggestions"""
        try:
            if viz_type == 'bar':
                self._create_enhanced_bar_chart(df, "Basic Query", {'type': 'bar'})
            elif viz_type == 'line':
                self._create_enhanced_line_chart(df, "Basic Query", {'type': 'line'})
            elif viz_type == 'scatter':
                self._create_enhanced_scatter_plot(df, "Basic Query", {'type': 'scatter'})
            elif viz_type == 'pie':
                self._create_enhanced_pie_chart(df, "Basic Query", {'type': 'pie'})
            elif viz_type == 'histogram':
                self._create_enhanced_histogram(df, "Basic Query", {'type': 'histogram'})
            else:
                self._create_enhanced_table(df, "Basic Query", {'type': 'table'})
        except Exception as e:
            st.error(f"❌ Error creating {viz_type} visualization: {str(e)}")
            self._create_enhanced_table(df, "Basic Query", {'type': 'table'})
    
    def create_custom_chart(self, df: pd.DataFrame, chart_type: str, 
                          x_col: str = None, y_col: str = None) -> None:
        """Create custom chart with user-specified parameters"""
        try:
            if chart_type.lower() == 'bar' and x_col and y_col:
                fig = px.bar(
                    df, 
                    x=x_col, 
                    y=y_col, 
                    title=f"{y_col} by {x_col}",
                    color_discrete_sequence=self.current_palette
                )
            elif chart_type.lower() == 'line' and x_col and y_col:
                fig = px.line(
                    df, 
                    x=x_col, 
                    y=y_col, 
                    title=f"{y_col} over {x_col}", 
                    markers=True,
                    color_discrete_sequence=self.current_palette
                )
            elif chart_type.lower() == 'scatter' and x_col and y_col:
                fig = px.scatter(
                    df, 
                    x=x_col, 
                    y=y_col, 
                    title=f"{y_col} vs {x_col}",
                    color_discrete_sequence=self.current_palette
                )
            else:
                st.error("❌ Invalid chart configuration. Please check your selections.")
                return
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show basic stats
            col1, col2 = st.columns(2)
            with col1:
                if y_col in df.select_dtypes(include=['number']).columns:
                    st.metric(f"Average {y_col}", f"{df[y_col].mean():.2f}")
            with col2:
                st.metric("Data Points", len(df))
            
        except Exception as e:
            st.error(f"❌ Error creating custom chart: {str(e)}")
            st.info("💡 Please check your column selections and data types.")