### Before running this project, you need to ensure that you have completed the following tasks:
- First, you need to understand **the purpose of each folder in the root directory**. All documentation **related to the runtime results** is located in the folders mentioned below (e.g., `data`, `origin_data`).
- Second, the folders `save_温度` and `save_湿度` contain **pre-trained models**. Before running the code, you need to copy the relevant model (e.g., temperature) to the `save` folder.
- Additionally, `LSTM.py`, `RNN.py`, and `GRU.py` are three prediction models, each containing **baseline models and FB models**; `selectPolicy.py` is our selection strategy **(FB)**, `plot_utils.py` contains **plotting** functions, and `prime.py` is used to plot the **original data graph**.
- Finally, you only need to specify the **data file path** in the `__init__` section of the **prediction model** file and **adjust the relevant parameters** to run the model for this project.
