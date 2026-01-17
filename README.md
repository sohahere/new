🎓 Student 360°: Holistic Performance & Well-being AIOverviewStudent 360° is an AI-driven intervention system designed to identify students at risk of academic failure before it happens. Unlike traditional models that look only at grades, 
this system utilizes a Dual-Pipeline Architecture:Quantitative Analysis: Evaluates academic metrics, study habits, and sleep patterns.
Qualitative Analysis (NLP): Analyzes daily text logs to detect hidden stress and burnout.The goal is not only to predict failure, but also to provide actionable counseling strategies tailored to the specific intersection of a student's performance and mental health.
🚀 Key Features1. Predictive Risk ModelingEngine: Logistic Regression with Hyperparameter Tuning (GridSearch).Threshold Tuning: Optimized via Precision-Recall curves to balance the trade-off between missing at-risk students and false alarms.
Feature Engineering: Custom metrics like Academic Risk Score, Effort Score, and Sleep Deviation.2. NLP Stress DetectionTechnique: TF-IDF Vectorization combined with classification.
Input: Daily student text logs.Output: Classifies high-stress vs. low-stress states to contextualize academic performance.3. The "What-If" SimulatorAn interactive engine that simulates behavioral changes.Example: "If Student X increases library hoursby 2 hours and reduces social media by 1 hour, does their risk drop to the 'Safe Zone'?
"4. Holistic Counseling Engine (The 4 Quadrants)Merges Academic Risk and Stress levels to categorize students into actionable intervention groups:🛑 High Functioning Burnout: (Good Grades + High Stress) -> Needs Rest.🚨 Crisis Mode: (Poor Grades + High Stress) -> Needs Mental Stabilization.💤 Disengaged: (Poor Grades + Low Stress) -> Needs Accountability.
🌟 Thriving: (Good Grades + Low Stress) -> Needs Enrichment.🛠️ ArchitectureThe system uses a hybrid approach to data processing:Code snippetgraph TD
    A[Raw Student Data] --> B{Data Type?}
    
    B -- Numerical --> C[Feature Engineering]
    C --> D[Standard Scaler]
    D --> E[Logistic Regression Model]
    E --> F[Academic Risk Probability]
    
    B -- Text Logs --> G[Text Cleaning & Regex]
    G --> H[TF-IDF Vectorization]
    H --> I[NLP Stress Model]
    I --> J[Stress Probability]
    
    F --> K[Decision Engine]
    J --> K
    
    K --> L[Final Diagnosis & Report]
📊 Methodology & Tech StackLibraries: Pandas, Numpy, Scikit-Learn, SHAP, Seaborn, Matplotlib1. Feature EngineeringWe transform raw data into behavioral insights:Sleep Deviation: $| \text{Sleep}_{\text{avg}} - 7 |$ (Measures irregularity).Academic Strength: Composite of GPA and recent test scores.Backlog Impact: Boolean flag for carry-over subjects.2. Model Explainability (SHAP)We use SHAP (SHapley Additive exPlanations) to ensure the model is not a black box. This tells us why a specific student was flagged:Did their attendance drop hurt them?Is social media usage the primary driver of their risk?3. Evaluation MetricsF1 Score: The primary metric to balance Precision and Recall.Stability Gap: We monitor the difference between Train F1 and Test F1 to prevent overfitting.💻 UsagePrerequisites:Bashpip install pandas numpy shap scikit-learn matplotlib seaborn
Run the Analysis:The script is self-contained. Ensure your dataset is located at the path defined in load_and_process_data.Python# The script will:
# 1. Train the Baseline and Optimized models.
# 2. Run the NLP pipeline on text logs.
# 3. Output a "Premium Report" for specific student profiles.
Sample Output (Console):Plaintext🎓 COMPREHENSIVE STUDENT SUCCESS REPORT (ID: 42)
================================================================
RISK BAND: ⚠️ HIGH RISK (Probability: 88.4%)
RECOMMENDATION: Priority Academic Counseling.

🛑 DIAGNOSIS: ACADEMIC CRISIS MODE
Observation: Performance is dropping AND stress is peaking.
💊 Prescription: STOP focusing on the backlog. Focus on mental stabilization first.
🧠 Future ImprovementsTime-Series Analysis: Incorporate LSTM models to track risk velocity (how fast is a student dropping?) over the semester.Intervention Tracking: Add a feedback loop to measure if the "Prescribed" counseling actually improved the student's next test score
