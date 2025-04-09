
# 🍁FutureCanada: Forecasting Canada's Growth & Infrastructure Capacity

**Future Canada** is an intelligent forecasting tool designed to help policymakers, analysts, and researchers assess **Canada’s future population absorption capacity**. Powered by predictive models and interactive dashboards, the project simulates future scenarios to highlight potential pressures on **Housing**, **Education**, and **Healthcare** infrastructure across provinces.

**In collaboration with:**

![IRCC Logo](https://github.com/user-attachments/assets/1a8282ba-02e8-4d45-995b-87755eadc4c6)


---

## ✨ Key Features

- 🔄 **Automated Data Cleaning**: Upload raw data and let the system clean and format it automatically.
- 📈 **Population Forecasting**: Uses demographic models based on birth, death, and immigration data.
- 🏠 🏥 🎓 **Infrastructure Forecasting**: Projects needs in Housing, Healthcare, and Education.
- 🎛️ **Power BI Dashboard**: Adjust TR (Total Rate) and PR (Per Capita Rate) to simulate real-world scenarios.
- 📍 **Province-Level Customization**: Analyze results province-wise to guide targeted policy decisions.

---

## 📦 Dependencies

Ensure you have the following Python packages installed:

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 🛠️ How to Use

> The tool is fully modular and automated. No manual processing required.

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/future-canada.git
   cd future-canada
   ```

2. **Upload your raw files** into the `/data/raw/` directory.

3. **Run the main script** to clean and forecast the data:
   ```bash
   python main.py
   ```

4. **Open Power BI report** from the `/dashboard` directory and load the generated forecast data:
   - File: `FutureCanadaDashboard.pbix`

5. **Adjust TR and PR** in Power BI to see how infrastructure and education demand changes across provinces.

---

## 📁 Project Structure

```
future-canada/
│
├── data/
│   ├── raw/               # Upload raw datasets
│   ├── processed/         # Cleaned and structured data
│   └── forecasted/        # Output forecasts for Power BI
│
├── models/                # Model evaluation and storage
│   └── housing_model_evaluation_metrics.csv
│
├── src/
│   ├── data_demographics/ # Scripts for demographic data
│   ├── data_processing/   # Cleaning and transformation logic
│   ├── features/          # Feature engineering utilities
│   └── machine_learning/  # Forecasting and model building
│
├── main.py                # Master execution script
└── README.md
```

---

## 📊 Power BI Dashboard Preview

Use the sliders for TR (Temporary Residents) and PR (Permenant Residents) to interactively model impacts on infrastructure.

![image](https://github.com/user-attachments/assets/4d1d10d9-1a2b-4de1-b3dd-14dec2d75907)
![video](https://www.loom.com/share/39b336bde3f54e4b8aef99836f06c2e6?sid=bd93d4d9-87cb-4ab5-98c5-f9caa7393996)


---

## 💡 Use Cases

- **Policy Planning**: Identify infrastructure investment needs across provinces.
- **Scenario Simulation**: Test effects of varying immigration and birth/death rates.
- **Research & Education**: Understand Canada’s future demographic challenges.

---

## 🔮 Future Scope

- Connect with **live IRCC immigration feeds** for real-time forecasting.
- Add **economic and environmental** data dimensions.
- Enable drill-down to **city or regional level** forecasting.

---

## 📬 Contact

For inquiries, collaboration, or suggestions:

- 📧 soni0050@algonquinlive.com
- 🔗 [LinkedIn Profile](https://www.linkedin.com)
- 🌐 [Project Website or Portfolio](https://www.yourwebsite.com)
