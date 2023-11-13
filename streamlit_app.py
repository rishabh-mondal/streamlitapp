import numpy as np

# import torch
# import hamiltorch
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import streamlit as st
import torch.nn as nn
import seaborn as sns
import torch.nn.functional as F

st.set_option("deprecation.showPyplotGlobalUse", False)


X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

st.subheader("Make Moons Dataset")
scatter_plot = sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Set1")
scatter_plot.set_title("Make Moons Dataset")
scatter_plot.set(xlabel="X1", ylabel="X2")
st.pyplot()
