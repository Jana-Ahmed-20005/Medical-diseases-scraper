# 🏥 Medical Project - Intelligent Disease Visualization and Analysis Tool

## 🔬 Overview

The **Medical Project** is a Tkinter-based desktop application that combines web scraping, network science, data visualization, and medical intelligence to extract, analyze, and visualize disease information. It integrates advanced techniques such as:

- Web scraping from trusted sources (e.g., Hugging Face)
- Graph-based network construction and analysis
- Heatmap generation and 3D modeling
- Informative visualizations using matplotlib
- Animated splash screen using OpenCV

---

## 📌 Features

### 1. 🎬 Splash Screen
- Displays a welcoming video (`startup video.mp4`) using OpenCV and Tkinter canvas.

### 2. 🌐 Web Scraping Tab
- Extracts datasets related to diseases and symptoms using SerpAPI from sources like Hugging Face.
- Parses relevant information (name, symptoms, treatment) using BeautifulSoup.
- Displays the raw and processed information in a Tkinter text widget.

### 3. 🔗 Network Construction Tab
- Builds an undirected graph of diseases based on shared symptoms and treatments using NetworkX.
- Displays the disease-disease connection graph interactively.

### 4. 📊 Network Analysis Tab
- Computes:
  - Node degrees
  - Betweenness centrality (normalized and unnormalized)
- Identifies key diseases based on centrality metrics.

### 5. 🌡️ Heatmap Tab
- Simulates geospatial disease intensity data using a custom kernel density estimation (KDE).
- Overlays the heatmap on a silhouette image of the human body.

### 6. 🧬 3D Model Tab
- Renders a 3D network graph of disease relationships using `matplotlib`'s 3D capabilities.

### 7. 📈 Bonus Graphs Tab
- Displays additional visualizations such as:
  - Bar chart of number of symptoms per disease
  - Histogram showing distribution of symptom counts

---

## 🛠️ Technologies Used

| Category         | Tools & Libraries               |
|------------------|----------------------------------|
| GUI Framework     | `tkinter`, `ttk`                 |
| Data Handling     | `requests`, `bs4`, `serpapi`     |
| Visualization     | `matplotlib`, `networkx`, `cv2`, `Pillow`, `numpy` |
| Video Processing  | `OpenCV`                         |
| 3D Graph Modeling | `matplotlib` 3D projection       |
| Image Handling    | `Pillow`, `OpenCV`               |

---

## 🔧 Setup Instructions

### 📦 Prerequisites

Ensure Python 3.7+ is installed. Install required libraries using:

```bash
pip install pillow matplotlib numpy opencv-python networkx serpapi beautifulsoup4

```
## 🔑 SerpAPI Key

Replace the placeholder in the code with your SerpAPI key:
```bash
api_key = 'YOUR_API_KEY'
```

##📂 Project Files Required
Make sure the following files are present in the root directory:

`startup video.mp4` – Intro video for the splash screen

`background.jpg` – Background for main page

`AI-web-scraping-.jpg`, `network.jpg`, `network analysis.jpg`, `heatmap.jpg`, `3d model.jpg` – Tab-specific backgrounds

`Human_body_silhouette.png` – Used for heatmap overlay


## 🚀 How to Run
```bash
python your_script_name.py
```
The application will launch with a splash screen and transition to a multi-tabbed medical interface.


## 🧩 Future Enhancements
- Integration with live health data sources (e.g., WHO, CDC)
- Exportable reports for medical insights
- Enhanced UI with dark/light themes
- More granular heatmap based on real geolocation data
