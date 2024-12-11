# NLBSE-25: Enhancing Multi-Label Code Comment Classification

Welcome to the repository for **"Enhancing Multi-Label Code Comment Classification with Data Augmentation and Transformer-Based Architectures"**, submitted as a solution for the NLBSE'25 Code Comment Classification Tool Competition. This repository hosts all relevant scripts, datasets, and model configurations used in the project.

---

## Abstract

Code comment classification is vital for software comprehension and maintenance. This repository demonstrates a multi-step solution that achieves a **6.7% accuracy improvement** over baseline models by combining **synthetic dataset generation** and **fine-tuned transformer-based models**.

Key Points:
- Translation-retranslation for **linguistic diversity** in data augmentation.
- Transformer architectures (**BERT, RoBERTa, CodeBERT, XLNet**) for multi-label classification.
- Tailored frameworks for Java, Python, and Pharo databases.

---

## Repository Structure

```plaintext
.
├── Dataset Generation/     # Scripts for data augmentation (translation-retranslation pipelines).
├── Datasets/               # Original, augmented, and filtered datasets.
├── HyperParameter tuning Python/ # Optuna-based hyperparameter optimization scripts run with python dataset to select the best model.
├── Model-Saving/           # Fine-tuned transformer models saved to huggingface.
├── roBERTa-large-hyperparameter-java-pharo/ # Scripts for RoBERTa tuning on specific languages.
└── final-score.ipynb       # Results, evaluation metrics.
```
## Note

All notebooks and scripts in this repository were executed in the Kaggle environment using **Kaggle T4 * 2 GPUs** and **P100 GPU**. If you plan to run the notebooks in Kaggle, the platform provides an ideal environment with pre-installed dependencies and powerful GPUs, requiring minimal setup.

---

### Example: Uploading a `.ipynb` File to Kaggle

1. **Go to Kaggle**:
   - Log in to your Kaggle account and navigate to the **Code** section.

2. **Create a New Notebook**:
   - Click on the **New Notebook** button in the top-right corner.

3. **Upload Your Notebook**:
   - In the new notebook interface, click the **File** dropdown menu in the top-left corner.
   - Select **Upload Notebook** from the dropdown.
   - Choose the `.ipynb` file from your local computer and upload it.

---

## How to Use (in Kaggle):

1. In the `Datasets` folder, there are three datasets. Go to Kaggle and use the **Create Dataset** option to upload these datasets as a Kaggle dataset.

2. Once uploaded, open any of the notebooks you want to run in Kaggle (e.g., `final-score.ipynb`).

3. In the Kaggle environment:
   - Navigate to the **Add Dataset** option on the right panel of the notebook interface.
   - Search for and add the dataset you uploaded in Step 1.

4. Ensure that the notebook is configured to use a Kaggle runtime with **T4 * 2 GPUs**:
   - Go to **Settings** in the notebook interface.
   - Enable **Accelerator** and select **GPU (T4)**.

5. **Notebook-Specific GPU Requirements**:
   - For `XLnet-base.ipynb` and `XLnet-large.ipynb`, use **GPU P100**:
     - Go to **Settings** in the notebook interface.
     - Select **GPU (P100)**.

6. **Running the Notebook**:
   - Execute the notebook cells sequentially to run the experiments.

7. **Setting Up Secrets for W&B and Hugging Face**:
   - Generate API keys from:
     - **Weights & Biases (W&B)**: Go to your W&B account settings > **API Keys**.
     - **Hugging Face**: Go to your Hugging Face account settings > **Access Tokens**.
   - In your Kaggle notebook:
     - Navigate to the **Add-ons** menu > **Secrets**.
     - Add the secrets for W&B and Hugging Face using the names specified in the notebook (e.g., `WANDB_API_KEY` and `HUGGINGFACE_API_KEY`).

8. **Adjust File Paths**:
   - Update the dataset file paths in the notebook. For example:
     ```python
     pd.read_csv('/kaggle/input/your-dataset-name/filename.csv')
     ```
   - Replace `/kaggle/input/your-dataset-name/filename.csv` with the actual path of your uploaded dataset in Kaggle.

**Note:** The Kaggle environment comes pre-installed with most dependencies. However, if you need additional packages, install them using the `!pip install` command in a notebook cell.


## How to Use (locally):

1. Download the repository:

   ```bash
   git clone https://github.com/Mushfiqur6087/NLBSE-25.git
   cd NLBSE-25.git
   
2. Install dependencies (if running locally):
   ```bash
   pip install -r requirements.txt

3. Adjust Input Paths
   - Update the file paths in the notebooks or scripts to match your local directory structure. 
   For example, replace:
   ```python
   pd.read_csv('/kaggle/input/your-dataset-name/filename.csv')
   ```
   with
   ```python
   pd.read_csv('path-to-your-local-dataset/filename.csv')
   ```
4. Remove wandb login.
   - Comment out
   ```python
   # import wandb
   # wandb.init()
   ``
