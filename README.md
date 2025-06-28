<div align="center">
  <!-- Neural network banner with name -->
  <img src="https://github.com/Ishmeet13/Ishmeet13/blob/9ae8b2fcb36b34e2302a924fcef5d30ef59a5d3c/Banner-img.png" alt="Ishmeet Singh Arora - Neural Network Banner" width="100%" height="180" />
  
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=700&size=28&duration=3000&pause=1000&color=000000&center=true&vCenter=true&width=435&lines=AI+Engineer;ML+Researcher" alt="Typing SVG" />
  
  ### AI Engineer & Machine Learning Researcher
  
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ishmeet-singh-arora-a91344200)
  [![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:arora9e@uwindsor.ca)
  [![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/Ishmeet13)
  
</div>

---

> *Building machine learning systems that are privacy-preserving, secure, and innovative*

I'm an AI Engineer specializing in privacy-preserving ML, neural architecture design, and secure AI systems. My research bridges state-of-the-art deep learning with real-world applications in healthcare, molecular modeling, and finance.

- 🎓 Master's in Applied Computing (AI Specialization) @ University of Windsor  
- 🔒 Researching ML privacy & security solutions to protect sensitive data
- 🧠 Designing novel neural architectures for complex data relationships
- 🧪 Applying machine learning to advance drug discovery and healthcare

---

## 🔬 Research Focus

|   Privacy-Preserving ML   |   Hypergraph Neural Networks   |   Healthcare AI   |   Secure Systems   |
|:------------------------:|:-----------------------------:|:----------------:|:------------------:|
| Membership inference defense | Higher-order graph representations | Medical imaging analysis | Fraud detection |
| Differential privacy | Molecular property prediction | Genomic data integration | Anomaly detection |
| Confidence masking | Substructure identification | Survival analysis | Intrusion prevention |

---

## 🚀 Key Projects

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

### 🐍 Customer Satisfaction Predictor

*Building production-ready ML pipelines with ZenML and MLflow*

- End-to-end ML pipeline for customer satisfaction prediction on e-commerce data
- Automated training, evaluation, and deployment workflows
- Interactive Streamlit interface for real-time predictions
- **Tech Stack:** ZenML, MLflow, Streamlit, Python

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

### 🧬 HyperMolecule: Drug Discovery Engine

*Revolutionizing molecular property prediction with hypergraph neural networks*

- Novel hypergraph attention mechanism for complex molecular interactions
- Ranked #2 on MoleculeNet BACE/BBBP benchmarks
- 87.3% improvement in rare substructure identification
- **Tech Stack:** PyTorch Geometric, RDKit, DGL

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

### 🔍 ProcessTree Explorer

*System programming utility for Linux process management*

- Maps parent-child relationships in the process hierarchy
- Identifies zombie processes and orphaned process chains
- Controls process trees with custom signal management
- **Tech Stack:** C, Linux /proc filesystem, Signal handling

### 🏠 House Price Prediction System

*Complete ML pipeline for real estate valuation*

- Developed predictive analytics for property pricing using extensive feature engineering
- End-to-end pipeline from data cleaning to model deployment
- Achieved 93.7% accuracy on regional housing market data
- **Tech Stack:** scikit-learn, XGBoost, Pandas, Matplotlib

### 💳 Credit Card Advisor

*Java application using Selenium for financial product comparison*

- Web scraping system for credit card data collection from banking websites
- Spring Boot GUI for matching user preferences with optimal card options
- Interactive comparison system with financial recommendation engine
- **Tech Stack:** Java, Selenium, Spring Boot, Web Scraping

### 🧬 Breast Cancer Genomic Analysis

*ML-based analysis of cancer genomic data from Memorial Sloan-Kettering*

- Predictive models for survival based on mutation profiles
- Random Forest (85% accuracy), SVM (83%), and Naive Bayes implementation
- Identified key genetic markers in cancer progression
- **Tech Stack:** Python, scikit-learn, Pandas, Matplotlib

### 🔒 Membership Inference Attack

*Implementation of privacy vulnerability testing for ML models*

- Comprehensive evaluation of CNN models on standard datasets
- Demonstrated varying attack success rates based on model complexity
- Privacy vulnerability assessment framework for deep learning systems
- **Tech Stack:** PyTorch, LightGBM, NumPy, Privacy Analysis

---

## 🔬 Currently Working On

### 📊 Comparative Analysis of Hypergraph Neural Networks for Molecular Property Prediction

*Investigating the role of dense subgraph construction in molecular modeling*

- Evaluating different hypergraph construction techniques for molecule representation
- Benchmarking various neural architectures for property prediction tasks
- Analyzing the impact of dense subgraph identification on model performance
- Developing novel attention mechanisms for higher-order molecular interactions
- **Tech Stack:** PyTorch Geometric, DGL, RDKit, JAX

---

## 💻 Technical Skills

<div align="center">
  
  ### ML & Data Science
  ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
  ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
  ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
  ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
  ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
  ![JAX](https://img.shields.io/badge/JAX-0A66C2?style=flat&logo=jax&logoColor=white)
  
  ### Development & DevOps
  ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
  ![Java](https://img.shields.io/badge/Java-ED8B00?style=flat&logo=java&logoColor=white)
  ![C++](https://img.shields.io/badge/C++-00599C?style=flat&logo=c%2B%2B&logoColor=white)
  ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
  ![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=flat&logo=kubernetes&logoColor=white)
  
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
        <h3>🖥️ Software Developer Intern</h3>
        <p><em>iDesign.market, Delhi | Jan 2024 - Jun 2024</em></p>
        <ul>
          <li>Mastered WebRTC for advanced real-time communication features</li>
          <li>Engineered video chat functionality with instant messaging capabilities</li>
          <li>Developed intuitive project management interfaces for stakeholder insights</li>
        </ul>
      </td>
      <td>
        <h3>🔐 Security Analyst Intern</h3>
        <p><em>Acmegrade, Bangalore | Mar 2023 - Aug 2023</em></p>
        <ul>
          <li>Conducted network vulnerability assessments using Nmap Script Engines</li>
          <li>Performed database security analysis with SQL Map</li>
          <li>Strengthened data protection through vulnerability identification and remediation</li>
        </ul>
      </td>
    </tr>
  </table>
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
