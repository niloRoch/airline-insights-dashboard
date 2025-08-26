# Airlines Analysis Dashboard ✈️

A comprehensive, interactive dashboard for analyzing airline flight data with advanced visualizations, market segmentation, and intelligent recommendations.

## 📁 Project Structure

```
airlines-analysis-dashboard/
├── 📄 main.py                    # Main Streamlit app
├── 📂 pages/                     # Multi-page dashboard
│   ├── 1_📊_Overview.py         # Executive overview and KPIs
│   ├── 2_🏢_Airlines_Analysis.py # Detailed airline performance
│   ├── 3_⏰_Temporal_Analysis.py  # Time-based patterns analysis
│   ├── 4_🎯_Market_Segmentation.py # Customer clustering
│   └── 5_💡_Recommendations.py   # AI-powered recommendations
├── 📄 requirements.txt           # Python dependencies
├── 📂 .streamlit/
│   └── config.toml              # Streamlit configuration
└── 📂 data/                     # Data directories
    ├── raw/                     # Raw data files
    └── processed/               # Processed data files
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd airlines-analysis-dashboard
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Your Data
Place your airline data CSV file in the `data/raw/` directory with the name `airlines_flights_data.csv`.

**Required columns:**
- `airline` - Airline name
- `price` - Flight price
- `duration` - Flight duration in hours
- `departure_time` - Time slot (Morning, Afternoon, Evening, etc.)
- `stops` - Number of stops (use 'zero' for direct flights)

### 4. Run the Dashboard
```bash
streamlit run main.py
```

The dashboard will be available at `http://localhost:8501`

## 📊 Dashboard Features

### 🏠 Main Dashboard (main.py)
- Executive KPI metrics
- Market overview charts
- Real-time dashboard status
- Quick insights and summary statistics

### 📊 Overview Analysis
- Comprehensive KPI dashboard
- Price distribution analysis with statistical overlays
- Market share visualization
- Correlation analysis
- Data quality assessment

### 🏢 Airlines Analysis
- Detailed airline performance metrics
- Competitive positioning analysis
- Price performance matrix
- Service quality indicators
- Interactive airline comparison tools

### ⏰ Temporal Analysis
- Price heatmaps by time slots
- Peak hours identification
- Temporal demand patterns
- Time-based efficiency analysis
- Best value time slot recommendations

### 🎯 Market Segmentation
- Advanced clustering analysis using K-means
- Customer segment profiling
- PCA visualization
- Optimal cluster determination
- Strategic recommendations by segment

### 💡 Intelligent Recommendations
- Multi-criteria recommendation system
- Personalized user profiles:
  - Budget Conscious
  - Time Sensitive
  - Convenience Seeker
  - Quality Focused
  - Balanced
- Smart filtering options
- Profile-based suggestions
- Export functionality

## 🔧 Configuration

### Streamlit Configuration
The app uses custom configuration in `.streamlit/config.toml`:
- Custom theme colors
- Performance optimizations
- Browser settings

### Data Requirements
The system can work with either:
1. **Processed data** - Enhanced dataset with all features
2. **Raw data** - Basic dataset (features will be auto-generated)

## 🎨 Key Features

### Advanced Analytics
- Statistical analysis with correlation matrices
- Machine learning clustering
- Multi-criteria decision analysis
- Predictive scoring algorithms

### Interactive Visualizations
- Plotly-based interactive charts
- Real-time filtering and updates
- Professional color schemes
- Mobile-responsive design

### Smart Recommendations
- AI-powered recommendation engine
- Personalization based on user profiles
- Dynamic scoring algorithms
- Export capabilities

### Performance Optimization
- Cached data loading
- Efficient computation
- Responsive user interface
- Fast visualization rendering

## 📈 Usage Examples

### For Business Analysts
- Monitor airline market share and performance
- Identify optimal pricing strategies
- Analyze customer segments
- Generate executive reports

### For Travelers
- Find best flight deals based on preferences
- Compare airlines and routes
- Get personalized recommendations
- Analyze price trends by time

### For Airline Operations
- Benchmark performance against competitors
- Optimize scheduling and pricing
- Identify market opportunities
- Improve service quality metrics

## 🛠️ Technical Details

### Built With
- **Streamlit** - Web application framework
- **Plotly** - Interactive visualizations
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms
- **NumPy** - Numerical computations

### Data Processing Pipeline
1. **Data Loading** - Flexible CSV import
2. **Feature Engineering** - Automated feature creation
3. **Statistical Analysis** - Comprehensive analytics
4. **Machine Learning** - Clustering and scoring
5. **Visualization** - Interactive dashboard creation

### Performance Features
- Caching for fast data access
- Optimized clustering algorithms
- Efficient visualization rendering
- Responsive UI components

## 📊 Sample Visualizations

The dashboard includes:
- Executive KPI cards with trend indicators
- Interactive price distribution histograms
- Market share pie charts with drill-down
- Correlation heatmaps
- Time series analysis charts
- 3D clustering visualizations
- Multi-criteria radar charts
- Recommendation scoring displays

## 🔧 Customization

### Adding New Features
1. Create new functions in the appropriate page file
2. Update visualization functions
3. Add new filters or analysis options
4. Extend recommendation algorithms

### Styling
- Custom CSS in each page file
- Color palette configuration
- Responsive design elements
- Professional UI components

## 📝 Data Format

### Expected CSV Structure
```csv
airline,price,duration,departure_time,stops
Indigo,5000,2.5,Morning,zero
SpiceJet,4500,3.0,Evening,one
...
```

### Auto-Generated Features
- `efficiency_score` - Price per hour
- `is_direct` - Direct flight indicator
- `is_premium_time` - Premium time slot indicator
- `price_category` - Budget/Mid-Range/Premium
- `recommendation_score` - AI-generated score

## 🚀 Deployment

### Local Development
```bash
streamlit run main.py
```

### Production Deployment
- Streamlit Cloud
- Heroku
- AWS/GCP/Azure
- Docker containers

## 📞 Support

For technical issues or feature requests:
1. Check the requirements and data format
2. Verify all dependencies are installed
3. Ensure data files are in correct locations
4. Review error messages in terminal

## 🎯 Roadmap

Future enhancements:
- Real-time data integration
- Advanced ML models
- Mobile app version
- API endpoints
- Enhanced export options