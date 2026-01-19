<div align="center">
  <!-- Neural network banner with name -->
  <img src="https://github.com/Ishmeet13/Ishmeet13/blob/9ae8b2fcb36b34e2302a924fcef5d30ef59a5d3c/Banner-img.png" alt="Ishmeet Singh Arora - Neural Network Banner" width="100%" height="180" />
  
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=700&size=28&duration=3000&pause=1000&color=000000&center=true&vCenter=true&width=435&lines=AI+Engineer;ML+Researcher;Data+Scientist" alt="Typing SVG" />
  
  ### AI Engineer & Machine Learning Researcher
  
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ishmeet-singh-arora-a91344200)
  [![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:isarora2003@gmail.com)
  [![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/Ishmeet13)
  
</div>

---

> *I see patterns where most people see noise. Turning messy, real-world data into models, dashboards, and stories that help teams make decisions they can actually feel confident about.*

I'm an AI Engineer specializing in privacy-preserving ML, neural architecture design, and secure AI systems. My work sits at the intersection of technical depth and clear communication—whether it's forecasting demand, building ML pipelines, or simplifying complex analytics for non-technical stakeholders. I focus on solutions that are practical, understandable, and tied to real business outcomes.

- 🎓 Master's in Applied Computing (AI Specialization) @ University of Windsor  
- 🔒 Researching ML privacy & security solutions to protect sensitive data
- 🧠 Designing novel neural architectures for complex data relationships
- 📊 Building end-to-end ML pipelines from data to deployment

---

## 🔬 Research Focus

|   Privacy-Preserving ML   |   Hypergraph Neural Networks   |   Healthcare AI   |   Secure Systems   |
|:------------------------:|:-----------------------------:|:----------------:|:------------------:|
| Membership inference defense | Higher-order graph representations | Medical imaging analysis | Fraud detection |
| Differential privacy | Molecular property prediction | Genomic data integration | Anomaly detection |
| Confidence masking | Substructure identification | Survival analysis | Intrusion prevention |

---

## 🔬 Currently Working On

### 🔐 Username Existence Check System

*Production-style username availability service demonstrating scalable system design*

<p align="center">
  <img src="https://github.com/Ishmeet13/BloomCacheShards/blob/main/username-existence-check-flowchart.svg" alt="Username Existence Check Flowchart" width="700"/>
</p>

- **Bloom Filter** for ultra-fast negative checks — eliminates ~99% of DB queries
- **Redis Cache** with O(1) lookups and automatic in-memory fallback
- **SHA-256 based sharding** for uniform distribution and horizontal scalability
- **FastAPI REST API** with batch checks and username suggestion engine
- **Tech Stack:** Python, FastAPI, Redis, Bloom Filters, SHA-256, pytest

---

## 🚀 Key Projects

### 🤖 RAG Pipeline for Technical Documentation

*Intelligent document retrieval system for field operations*

- Built Retrieval-Augmented Generation pipeline using LangChain and Pinecone
- Achieved 90% context precision improvement for technical manual access
- Reduced lookup time by 40% with LLM-based query routing
- Designed Markdown/JSON knowledge templates for standardized data retrieval
- **Tech Stack:** LangChain, Pinecone, Hugging Face, Python

### 🏭 End-to-End ML Pipeline (Microservices + Deployment)

*Production-ready machine learning infrastructure*

- Built modular microservices for data ingestion, pre-processing, training, and inference
- Developed authentication, training orchestration, and prediction API services
- Integrated lightweight UI for real-time predictions and automated evaluation reports
- Improved model delivery reliability by 30% with scalable deployment architecture
- **Tech Stack:** Python, Flask, Scikit-learn, Docker, REST API

```python
# Microservice architecture for ML pipeline
class MLPipelineOrchestrator:
    def __init__(self):
        self.data_service = DataIngestionService()
        self.training_service = ModelTrainingService()
        self.inference_service = PredictionAPI()
    
    def run_pipeline(self, config):
        data = self.data_service.ingest(config.source)
        model = self.training_service.train(data, config.params)
        return self.inference_service.deploy(model)
```

### 📈 Store-Item Demand Forecasting

*Retail time-series prediction at scale*

- Built store-item level demand prediction across 10 stores & 50 product categories
- Improved forecast accuracy by 22% using feature-engineered time-series data
- Engineered seasonality, trend, and lag-based features for holiday/promotional periods
- Conducted cross-store analysis for inventory optimization and stock-out reduction
- **Tech Stack:** Python, LightGBM, Pandas, NumPy, Time-Series Modeling

### 🌡️ Time-Series Forecasting Suite

*Multi-model temporal analytics pipeline*

- Implemented ARIMA, Prophet, and Gradient Boosting models for pollution/weather data
- Improved long-range forecasting accuracy by 17%
- Built complete pipeline with stationarity checks, decomposition, and data leakage prevention
- Delivered visualization dashboards for trends, seasonality, and residual patterns
- **Tech Stack:** Python, Statsmodels, Scikit-learn, Matplotlib

### 🛡️ ML-Guard: Privacy Defense Framework

*Protecting machine learning models from inference attacks*

- Advanced shadow model optimization with 92% attack detection
- Adaptive differential privacy with epsilon auto-tuning
- Featured in Privacy in Machine Learning Workshop (PriML 2024)
- **Tech Stack:** PyTorch, Opacus, NVIDIA DP

```python
def secure_inference(model, input_data, epsilon=2.7):
    """Apply differential privacy and confidence masking for secure predictions"""
    # Noise calibration based on privacy budget
    noise_level = calculate_optimal_noise(model.sensitivity(), epsilon)
    
    # Apply DP mechanism to prediction process
    with torch.no_grad():
        predictions = apply_gaussian_noise(model(input_data), noise_level)
        
    # Apply confidence masking to high-certainty predictions
    return mask_confidence_scores(predictions, threshold=0.95)
```

### 🧬 Breast Cancer Genomic Analysis

*ML-based survival prediction from Memorial Sloan-Kettering data*

- Built survival-prediction pipeline using mutation and clinical data from 1,918 patients
- Achieved 85% accuracy (AUC ≈ 0.87) using Random Forest and SVM models
- Engineered mutation-based biomarkers (TP53, PIK3CA) for feature selection
- Identified mutation clusters improving risk-stratification for clinical decision-making
- **Tech Stack:** Python, Scikit-learn, Pandas, Cancer Genomics Data

### 🧬 HyperMolecule: Drug Discovery Engine

*Hypergraph neural networks for molecular property prediction*

- Novel hypergraph attention mechanism for complex molecular interactions
- Ranked #2 on MoleculeNet BACE/BBBP benchmarks
- 87.3% improvement in rare substructure identification
- Processed 7k+ molecular graphs; ranked top 5 among 70+ university projects
- **Tech Stack:** PyTorch Geometric, RDKit, DGL, AUC-ROC

```python
class HypergraphConv(nn.Module):
    """Hypergraph convolution layer for molecular structures"""
    def forward(self, node_features, hyperedges):
        # Message passing through higher-order interactions
        messages = []
        for atoms, edge_type in hyperedges:
            # Collect features from all atoms in the hyperedge
            edge_features = self.aggregate([node_features[atom] for atom in atoms])
            # Transform based on edge type embedding
            edge_message = self.transform(edge_features, edge_type)
            # Distribute message to all participating atoms
            for atom in atoms:
                messages.append((atom, edge_message))
        
        # Update atom representations
        return self.update_nodes(node_features, messages)
```

### 🖥️ Distributed File System with Multi-Server Architecture

*Specialized servers handling different file types with seamless management*

- Automatic file routing based on extensions to appropriate specialized servers
- Comprehensive file operations across distributed storage nodes
- Advanced path resolution and parallel processing capabilities
- **Tech Stack:** C, Sockets, Process Management, Network Programming

### 🐚 Unix-Style Shell Implementation

*Lightweight shell with advanced piping and process management*

- Supports standard and reverse piping, I/O redirection
- Implements conditional (&&, ||) and sequential command execution
- Custom file operations and robust error handling
- **Tech Stack:** C, Process Management, File I/O, Signal Handling

---

## 💻 Technical Skills

<div align="center">
  
  ### ML & Data Science
  ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
  ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
  ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
  ![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat&logo=lightgbm&logoColor=white)
  ![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat&logo=xgboost&logoColor=white)
  ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
  ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
  
  ### Data Engineering & Analytics
  ![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=flat&logo=powerbi&logoColor=black)
  ![Tableau](https://img.shields.io/badge/Tableau-E97627?style=flat&logo=tableau&logoColor=white)
  ![Airflow](https://img.shields.io/badge/Airflow-017CEE?style=flat&logo=apacheairflow&logoColor=white)
  ![dbt](https://img.shields.io/badge/dbt-FF694B?style=flat&logo=dbt&logoColor=white)
  ![Snowflake](https://img.shields.io/badge/Snowflake-29B5E8?style=flat&logo=snowflake&logoColor=white)
  
  ### Development & DevOps
  ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
  ![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat&logo=postgresql&logoColor=white)
  ![R](https://img.shields.io/badge/R-276DC3?style=flat&logo=r&logoColor=white)
  ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
  ![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white)
  
  ### Cloud & AI Platforms
  ![Azure](https://img.shields.io/badge/Azure-0078D4?style=flat&logo=microsoftazure&logoColor=white)
  ![LangChain](https://img.shields.io/badge/LangChain-121212?style=flat&logo=chainlink&logoColor=white)
  ![Pinecone](https://img.shields.io/badge/Pinecone-000000?style=flat&logo=pinecone&logoColor=white)
  ![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?style=flat&logo=huggingface&logoColor=black)
  
  ### Web & API
  ![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)
  ![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
  ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
  
</div>

---

## 💼 Professional Experience

<div align="center">
  <table>
    <tr>
      <td>
        <h3>🤖 AI Intern</h3>
        <p><em>Scelta | Windsor, Ontario | May 2025 - Aug 2025</em></p>
        <ul>
          <li>Built RAG pipeline using LangChain & Pinecone — 90% context precision improvement</li>
          <li>Reduced technical manual lookup time by 40% with LLM-based query routing</li>
          <li>Designed Markdown/JSON knowledge templates for scalable documentation retrieval</li>
        </ul>
        <p><code>LangChain</code> <code>Pinecone</code> <code>Hugging Face</code> <code>Python</code></p>
      </td>
    </tr>
    <tr>
      <td>
        <h3>🖥️ Software Developer Intern</h3>
        <p><em>iDesign.market | Delhi, India | Jan 2024 - Jun 2024</em></p>
        <ul>
          <li>Built real-time chat system using React & REST APIs — 38% user engagement increase</li>
          <li>Reduced latency by 25% through React lifecycle optimization</li>
          <li>Improved sprint velocity by 15% with optimized Git workflows in Agile environment</li>
        </ul>
        <p><code>React</code> <code>REST APIs</code> <code>Git</code> <code>Agile/Scrum</code></p>
      </td>
    </tr>
    <tr>
      <td>
        <h3>📊 Data Analyst Intern</h3>
        <p><em>Acmegrade Pvt. Ltd. | Bengaluru, India | Mar 2023 - Aug 2023</em></p>
        <ul>
          <li>Built ETL and data validation scripts — 20% accuracy improvement, 30% fewer pipeline errors</li>
          <li>Developed interactive Power BI dashboards with DAX for operational KPI visibility</li>
          <li>Performed clustering analysis to identify underperforming regions for targeted marketing</li>
        </ul>
        <p><code>SQL</code> <code>Python</code> <code>Power BI</code> <code>DAX</code> <code>Clustering</code></p>
      </td>
    </tr>
  </table>
</div>

---

## 📜 Certifications

<div align="center">

| Certification | Issuer | Date |
|:-------------:|:------:|:----:|
| **Microsoft Power BI Data Analyst (PL-300)** | Microsoft | Jan 2025 |
| **Python Essentials for MLOps** | Duke University | Nov 2024 |
| **Google Data Analytics** | Coursera | Oct 2024 |

</div>

---

## 📝 Recent Research

- **Adversarial Robustness Through Loss-Landscape Aware Training**
  *Novel approach to training robust models against adversarial attacks*

- **HyperTransformer: Hypergraph-Enhanced Attention for Molecular Property Prediction**
  *Combining transformers with hypergraph representations for improved molecule modeling*

- **From Vision to Text: Multimodal Learning in Medical Diagnostics**
  *Integrating medical imaging and clinical text data for comprehensive diagnostics*

- **Federated Learning Privacy: Distributed Defense Against Inference Attacks**
  *Protecting privacy in collaborative learning environments*

---

<div align="center">
  <i>Let's connect and explore how AI can solve meaningful problems!</i>
  <br><br>
  <img src="https://komarev.com/ghpvc/?username=Ishmeet13&color=00FFB3" alt="Profile views"/>
</div>
