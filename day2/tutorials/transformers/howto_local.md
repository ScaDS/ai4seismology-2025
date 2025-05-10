# How to Set Up a Local Conda Environment and Run The Notebooks

Follow these steps to set up a local Conda environment with the provided file `conda-tsa-env.yml`, to start a local Jupyter Hub instance, and run the notebooks in this folder.

## Step 1: Install Conda
Ensure you have Conda installed on your system. You can download and install Miniconda (preferred) or Anaconda from their official website:
- [Miniconda](https://www.anaconda.com/download/success#miniconda)
- [Anaconda](https://www.anaconda.com/download/success)

## Step 2: Create the Conda Environment
1. Open a terminal or command prompt.
2. Navigate to this folder containing the `conda-tsa-env.yml` file:
    ```bash
    cd /path/to/ai4seismology-2025/day2/transformers
    ```
3. [Create the Conda environment](https://docs.conda.io/projects/conda/en/latest/commands/env/create.html) using the `conda-tsa-env.yml` file:
    ```bash
    conda env create -f conda-tsa-env.yml
    ```
4. Verify the environment was created successfully and is listed:
    ```bash
    conda env list
    ```

## Step 3: Activate the Conda Environment
Activate the newly created environment:
```bash
conda activate tsa-transformer-env
```

## Step 4: Run Jupyter Lab and the Notebooks
1. Ensure you are in the correct folder:
    ```bash
    cd /path/to/ai4seismology-2025/day2/transformers
    ```
2. Launch a local Jupyter Lab instance in your browser:
    ```bash
    jupyter lab
    ```
3. Open and run the notebooks from the home directory in Jupyter Lab:
    - `day2.1_transformer-tsa_1.ipynb`
    - `day2.1_transformer-tsa_2.ipynb`

## Step 5: Deactivate the Environment (Optional)
When you're done, deactivate the Conda environment:
```bash
conda deactivate
```

That's it! You have successfully set up the Conda environment and run the notebooks.
