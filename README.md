# AI Engineering Notebooks

A collection of nine end-to-end notebooks that follow one workflow: raw data, a working model or pipeline, then an honest look at where it breaks. They cover data cleaning, classical ML, deep learning, and LLM/RAG systems.

**Stack:** Python · pandas · NumPy · scikit-learn · XGBoost · PyTorch · sentence-transformers · ChromaDB · Gemini API · Google Colab

## Contents

### Data & classical ML
| Notebook | What it does | Key techniques |
|---|---|---|
| [01 Customer data cleaning](notebooks/01_customer_data_cleaning.ipynb) | Cleans messy customer and transaction tables, then joins them to flag transactions with no customer profile | custom `apply` parsers, category standardization, group-median imputation, left-merge with indicator |
| [03 Apartment price regression](notebooks/03_apartment_price_regression.ipynb) | Predicts nightly rental price from listing features | `ColumnTransformer` + `Pipeline`, `LinearRegression`, MAE / RMSE / R² |
| [05 Credit default: business impact](notebooks/05_credit_default_business_impact.ipynb) | Chooses decision thresholds for two default models by money, not accuracy | confusion-matrix cost model, threshold sweep |
| [06 Premium subscription ensembles](notebooks/06_premium_subscription_ensembles.ipynb) | Random Forest vs. XGBoost on behavior data; diagnoses overfitting from the train/test gap and tunes it down | bias/variance diagnosis, `learning_rate` / `max_depth` / `subsample` tuning |

### Deep learning
| Notebook | What it does | Key techniques |
|---|---|---|
| [08 Stress detection MLP](notebooks/08_stress_detection_pytorch_mlp.ipynb) | PyTorch MLP classifying stress from wearable sensor data | `DataLoader`, `BCEWithLogitsLoss`, Adam, train/val loss tracking |

### LLM & RAG
| Notebook | What it does | Key techniques |
|---|---|---|
| [10 LLM API log analysis](notebooks/10_llm_api_log_analysis.ipynb) | Finds why API cost and hallucination rates are high | tokens-per-character by language, temperature vs. hallucination score, internal-data dependence |
| [11 Structured extraction](notebooks/11_llm_structured_extraction.ipynb) | Turns free-text complaints into JSON (name, product, category, urgency) | Gemini API, prompt design, JSON parsing to DataFrame |
| [13 Semantic search with ChromaDB](notebooks/13_chromadb_semantic_search.ipynb) | Embeds a knowledge base and searches it with metadata filters | sentence-transformers, ChromaDB, department and `is_active` filtering |
| [15 RAG evaluation](notebooks/15_rag_evaluation.ipynb) | Builds filtered retrieval context, validates LLM citations, scores retrieval | structured context, citation validation, Hit@2 from trace logs |

## Example result

In notebook 05, the best threshold for each model differs, and that changes the outcome:

| Model | Best threshold | Net business impact |
|---|---|---|
| A | 0.3 | Rp 556,000,000 |
| B | 0.1 | Rp 407,600,000 |

Both models turn net-negative above a threshold of 0.7, so threshold choice matters as much as model choice.

## Running the notebooks

Notebooks were written on Google Colab. Datasets are not included, so each notebook expects its CSV or JSONL file in the working directory. Notebook 11 reads a `GEMINI_API_KEY` from Colab secrets.
