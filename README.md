A web-based Machine Learning analytics dashboard that predicts car mileage (MPG) using Linear Regression, Ridge Regression, and Lasso Regression models trained on the Auto MPG dataset.
Built as an academic Machine Learning project to demonstrate regularization, model comparison, bias–variance tradeoff, and hyperparameter sensitivity through an interactive dashboard.
Disclaimer: This is an academic demonstration project. It is not intended for production decision-making or real-world financial predictions.
🔍 Features
| Feature                        | Description                                                  |
| ------------------------------ | ------------------------------------------------------------ |
| **Multiple Regression Models** | Linear, Ridge (L2 regularization), Lasso (L1 regularization) |
| **Interactive Alpha Slider**   | Users can adjust regularization strength (alpha)             |
| **Live Performance Metrics**   | R² Score, MAE, RMSE updated on input change                  |
| **Model Comparison Chart**     | Side-by-side visual comparison of model performance          |
| **Alpha vs R² Visualization**  | Shows effect of alpha on Ridge & Lasso performance           |
| **Residual Analysis Panel**    | Plots residuals for diagnosis of model fit                   |
| **Responsive App UI**          | Built using Streamlit with live feedback                     |

🏛 System Architecture
┌──────────────────────────────────┐
│        Frontend (Dashboard)       │
│       Streamlit UI Components     │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│          ML Logic & Dashboard     │
│   Model training & visualization  │
│  Linear / Ridge / Lasso scoring   │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│         Dataset Layer            │
│    Auto MPG dataset (cleaned)    │
└──────────────────────────────────┘

📁 Component Responsibilities

| Component                | Responsibility                                     |
| ------------------------ | -------------------------------------------------- |
| **Frontend (Streamlit)** | Dashboard UI, user inputs, chart rendering         |
| **ML Engine**            | Model training + evaluation (Linear, Ridge, Lasso) |
| **Visualization Module** | Generates interactive charts & residual plots      |
| **Dataset Processing**   | Handles imputation, scaling, one-hot encoding      |

🧠 Tech Stack

| Layer           | Technology                                    |
| --------------- | --------------------------------------------- |
| UI / Dashboard  | Python + Streamlit                            |
| Modeling        | scikit-learn (LinearRegression, Ridge, Lasso) |
| Data Processing | Pandas, NumPy                                 |
| Visualization   | Matplotlib                                    |
| Deployment      | Streamlit Cloud                               |

📂 Project Structure

car-mpg-regularization/
├── app.py                  # Main Streamlit dashboard
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python version for deployment
├── car-mpg.csv             # Dataset file
└── README.md

📊 Dataset Information

Dataset: Auto MPG (miles per gallon) — originally from the UCI Machine Learning Repository via StatLib
| Field        | Type       | Description              |
| ------------ | ---------- | ------------------------ |
| mpg          | Continuous | Fuel efficiency (target) |
| cylinders    | Integer    | Number of cylinders      |
| displacement | Continuous | Engine size (cu inches)  |
| horsepower   | Continuous | Engine power             |
| weight       | Continuous | Vehicle weight (lbs)     |
| acceleration | Continuous | 0–60 mph acceleration    |
| model year   | Integer    | Year of manufacture      |
| origin       | Integer    | Car origin (1,2,3)       |
| car_name     | String     | Car identifier           |

🧠 ML Model Summary

| Property       | Value                           |
| -------------- | ------------------------------- |
| Dataset        | Auto MPG dataset                |
| Algorithms     | Linear Regression, Ridge, Lasso |
| Regularization | L2 (Ridge), L1 (Lasso)          |
| Hyperparameter | Alpha (0.01 – 5.0)              |
| Scaling        | StandardScaler used on features |
| Evaluation     | R², MAE, RMSE                   |


📊 Model Performance Outputs

| Model             | R² Score | Notes                       |
| ----------------- | -------- | --------------------------- |
| Linear Regression | Varies   | Baseline, no regularization |
| Ridge Regression  | Varies   | Regularized with L2         |
| Lasso Regression  | Varies   | Regularized with L1         |


🛠 How to Run Locally

Prerequisites

Python 3.9+

pip

1. Clone the Repository
   git clone https://github.com/naveenkumar921995-cmd/car-mpg-regularization.git
cd car-mpg-regularization

2. Create Virtual Environment
Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1

3. Install Dependencies
pip install -r requirements.txt

4. Run Dashboard
streamlit run app.py

Visit: http://localhost:8501

🌍 Deployment

Hosted on Streamlit Cloud
Runtime: Python 3.11 (configured via runtime.txt)
First load may take 30–60 seconds due to free tier cold start.

Live App:
🔗 https://car-mpg-regularization-hkvf3drjhcc7tmjj7x3ms6.streamlit.app/

📈 Example Usage

Adjust the alpha slider to choose regularization strength.

Click Refresh / Recompute.

View updated R² Scores, Comparison chart, and Alpha sensitivity curve.

Use the Residual Plot to analyze model errors.

📘 Detailed Documentation (Suggested)

The following documentation can be added under a docs/ directory:

| Doc File            | Purpose                               |
| ------------------- | ------------------------------------- |
| `setup.md`          | Installation & troubleshooting        |
| `architecture.md`   | Architecture and data flow            |
| `model.md`          | Model details and evaluation          |
| `dataset.md`        | Dataset description and preprocessing |
| `alpha_analysis.md` | Alpha impact visualization logic      |
| `deployment.md`     | Deployment instructions               |

🔮 Future Improvements

Add ElasticNet model comparison

Add Cross-validation support

Allow custom dataset upload

Add downloadable performance reports

Convert into a Docker container

📜 License

Developed for academic and portfolio purposes.
