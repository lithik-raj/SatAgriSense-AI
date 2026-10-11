# 🛰️ SENTINEL-EARTH AI
### Geo-Agricultural Decision Platform

**Tagline:** Observe Earth. Understand Nature. Empower Decisions.

**Live Demo:** https://hack10project.vercel.app/  
**GitHub:** https://github.com/lithik-raj/SatAgriSense-AI

## 🌍 Overview
SENTINEL-EARTH AI is a geospatial AI platform designed to support agriculture and environmental monitoring using satellite imagery, vegetation analysis, and machine-learning-based decision support.

## 🎯 Problem Statement
Farmers and environmental analysts need reliable information about vegetation health, soil conditions, weather, and land-use changes. Traditional monitoring can be time-consuming and may lack a unified view of these factors.

## 💡 Proposed Solution
The platform brings geospatial visualization, NDVI analysis, environmental indicators, crop recommendations, and yield estimation together in one dashboard.

## ✨ Key Features
- 🛰️ **Satellite Visualization:** Interactive maps, location search, and satellite-layer interfaces.
- 🌿 **NDVI Analysis:** Evaluates vegetation greenness using satellite reflectance.
- 🌾 **Crop Recommendation:** Displays crop-suitability suggestions.
- 📊 **Yield Estimation:** Presents estimated agricultural yield.
- 🌡️ **Environmental Indicators:** Temperature, humidity, soil moisture, rainfall, and related indicators.
- 🌳 **Land-Cover Monitoring:** Supports exploration of vegetation, water bodies, forests, and urban areas.
- 📅 **Historical Comparison:** Interface for comparing observations across time.
- 🤖 **AI Agricultural Assistant:** Interface for agriculture and satellite-related questions.
- 📄 **Reports and History:** PDF-report and scan-history interfaces.

*Feature availability depends on the actual backend and data-source integrations.*

## 🧮 NDVI Formula

**NDVI = (NIR − Red) / (NIR + Red)**

- **NIR:** Near-infrared reflectance.
- **Red:** Red-band reflectance.

NDVI values range from −1 to +1. Higher positive values often indicate greener vegetation.

## 🏗️ System Workflow

1. Select a geographical location.
2. View available satellite imagery.
3. Analyze vegetation using NDVI.
4. Review soil and environmental indicators.
5. View crop recommendations and yield estimates.
6. Compare historical observations when available.
7. Review insights through the dashboard and AI assistant.

## 🧰 Technology Stack

- **Deployment:** Vercel
- **Satellite Data:** Sentinel satellite interfaces
- **Geospatial Analysis:** NDVI and land-cover visualization
- **Machine Learning:** Crop recommendation and yield-estimation interfaces
- **AI:** Hugging Face LLM interface
- **Storage:** SQLite-backed history interface
- **Visualization:** Interactive maps and 3D globe
- **Reporting:** PDF-report interface

*Verify exact frameworks, dependencies, and integrations against the source code.*

## 🌱 Applications
- Precision agriculture and crop planning
- Vegetation and land-use monitoring
- Environmental assessment
- Water-resource observation
- Location-based agricultural decision support

## ⚠️ Limitations
- Satellite data may be affected by clouds and acquisition delays.
- NDVI alone cannot diagnose crop disease or guarantee yield.
- Yield predictions depend on suitable training data and validation.
- Environmental readings require reliable data sources.
- Dashboard values may be illustrative unless verified against live APIs.

## 🚀 Future Enhancements
- Integrate verified Sentinel-2 and other satellite-data APIs.
- Add historical NDVI time-series and change detection.
- Improve crop recommendations using soil and weather datasets.
- Validate yield predictions with real agricultural records.
- Add weather forecasts, alerts, and IoT sensor integration.
- Support regional languages and detailed downloadable reports.

## ▶️ Try the Application

**Live Demo:** https://hack10project.vercel.app/

For local execution, follow the dependency installation and environment-variable instructions documented in the source repository.

## 📌 Project Information

- **Project:** SENTINEL-EARTH AI
- **Domain:** Geospatial AI, Remote Sensing, Machine Learning, Computer Vision, and Smart Agriculture
- **Primary Goal:** Support agricultural and environmental decisions through satellite-based insights.

*Built to explore how satellite intelligence and AI can support smarter agriculture and environmental monitoring.*
