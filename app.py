# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score

LOGGER = get_logger(__name__)


def main():
    st.set_page_config(
        page_title="BinaryClassifier",
        page_icon="👋",
    )

    st.write("# Binary Classification Web App")
    st.sidebar.title("Configuration")

    st.markdown(
        "Classifying mushrooms as either edible or poisonous 🍄"
    )

    st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv('mushrooms.csv')
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data
    
    @st.cache_data(persist=True)
    def split(df):
        y = df.type
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=0)
        return x_train, x_test, y_train, y_test
    
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            ConfusionMatrixDisplay(model,x_test, y_test, display_labels=class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            RocCurveDisplay(model,x_test, y_test)
            st.pyplot()

        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            PrecisionRecallDisplay(model, x_test, y_test)
            st.pyplot()

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)











    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)

if __name__ == "__main__":
    main()
