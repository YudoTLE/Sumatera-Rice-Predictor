# Sumatera Rice Predictor
Machine learning model that predicts rice production per harvest area (ton/ha) in Sumatra based on climate data, including rainfall, humidity, and temperature.

## Dataset and Units of Measurement
The dataset used for this project consists of historical rice production data (from 1993 to 2020) for 8 provinces in Sumatra: Nanggroe Aceh Darussalam (NAD), Sumatera Utara, Riau, Jambi, Sumatera Selatan, Bengkulu, and Lampung. It includes data on annual rice production and the corresponding harvest area (land area). Additionally, climate data on rainfall, humidity, and average temperature is obtained from BMKG (Indonesian Meteorological, Climatological, and Geophysical Agency).

### Unit of Measurement:
- **Rainfall**: Millimeters (mm) per month
- **Humidity**: Percentage (%)
- **Temperature**: Degrees Celsius (°C)
- **Production per Harvest Area**: Tons per hectare (ton/ha)

## Getting Started

### Prerequisites
Clone this repository to your local machine:
```bash
git clone https://github.com/YudoTLE/Sumatera-Rice-Predictor.git
cd Sumatera-Rice-Predictor
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```

## How to Use
To use the trained model for predictions, run the following command:
```bash
python3 predict.py
```