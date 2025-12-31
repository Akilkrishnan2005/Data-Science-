# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 12:22:00 2025

@author: akilk
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import  mean_squared_error,r2_score, accuracy_score, confusion_matrix, classification_report, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

data = None
target_column = None

# ---------------- MAIN WINDOW ---------------- #
root = tk.Tk()
root.title("DATA SCIENCE MODEL DEPLOYMENT")
root.geometry("1100x750")

# ---------------- FUNCTIONS ---------------- #

def upload_file():
    global data
    file_path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv"),("Excel files","*.xlsx")])
    if file_path:
        try:
            if file_path.endswith(".csv"):
                data = pd.read_csv(file_path)
            else:
                data = pd.read_excel(file_path)
            display_data()
            
            # Update target column combobox
            target_column_menu['values'] = list(data.columns)
            if len(data.columns) > 0:
                target_column_menu.current(0)

            messagebox.showinfo("Success", "Dataset loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load file:\n{e}")

def display_data():
    if data is not None:
        text_preview.delete('1.0', tk.END)
        text_preview.insert(tk.END, str(data.head()))

def get_target_column():
    global target_column
    target_column = target_column_var.get().strip()
    
    if not target_column:
        messagebox.showerror("Error", "Please select a target column.")
        return False

    
    if target_column not in data.columns:
        messagebox.showerror("Error", "Invalid Target Column Name!")
        return False
    return True

def clear_plot_frame():
    for widget in plot_frame.winfo_children():
        widget.destroy()

def train_model():
    global target_column
    if data is None:
        messagebox.showerror("Error", "Please upload a dataset first.")
        return
    
    task = task_var.get()
    algorithm = algorithm_var.get()

    if task in ["Regression", "Classification"]:
        if not get_target_column():
            return
    else:
        target_column = None

    df = data.copy()

    # Handle Missing Values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        action = messagebox.askquestion(
            "Missing Values Detected",
            f"Dataset contains {missing_count} missing values.\n\n"
            "Do you want to fill them automatically?\n\n"
            "Yes = Fill with Mean/Most Frequent\nNo = Drop missing rows"
        )
        if action == "yes":
            for col in df.columns:
                if df[col].dtype in ['int64','float64']:
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df.dropna(inplace=True)

    if task in ["Regression", "Classification"]:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df.copy()
        y = None

    # Encoding
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    if y is not None and y.dtype == "object":
        y = LabelEncoder().fit_transform(y.astype(str))

    if task in ["Regression", "Classification"]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    result_text.delete('1.0', tk.END)
    clear_plot_frame()

    try:
        if task == "Regression":
            if algorithm == "Linear Regression":
                model = LinearRegression()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                r2 = r2_score(y_test, preds)
                result_text.insert(tk.END, f"RMSE: {rmse:.2f}\nR² Score: {r2:.2f}")
                plot_regression_results(y_test, preds)

        elif task == "Classification":
            if algorithm == "Logistic Regression":
                model = LogisticRegression()
            elif algorithm == "Decision Tree":
                model = DecisionTreeClassifier()
            elif algorithm == "Random Forest":
                model = RandomForestClassifier()
            elif algorithm == "Naive Bayes":
                model = GaussianNB()
            elif algorithm == "KNN":
                model = KNeighborsClassifier()
            elif algorithm == "SVM":
                model = SVC(probability=True)
                
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            report = classification_report(y_test, preds)

            result_text.insert(tk.END, f"Accuracy: {acc:.2f}\n\n{report}")

# Confusion Matrix
            plot_confusion_matrix_tkinter(y_test, preds)

# Classification Decision Boundary (ONLY if 2 features)
            plot_classification_results(X_train, y_train, model)

'''
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            report = classification_report(y_test, preds)
            result_text.insert(tk.END, f"Accuracy: {acc:.2f}\n\n{report}")
            plot_confusion_matrix_tkinter(y_test, preds)
'''
        elif task == "Clustering":
            if algorithm == "KMeans":
                model = KMeans(n_clusters=3, random_state=42)
                preds = model.fit_predict(X)
                score = silhouette_score(X, preds)
                result_text.insert(tk.END, f"Silhouette Score: {score:.2f}\n")
                plot_clusters(X, preds)

    except Exception as e:
        messagebox.showerror("Training Error", str(e))


# ---------------- PLOTTING FUNCTIONS ---------------- #

def plot_regression_results(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)

    ax.scatter(y_true, y_pred, color="red", alpha=0.6)
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            color="blue", linewidth=2)

    ax.set_title("Regression: Actual vs Predicted")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.grid(True)

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    
    
def plot_classification_results(X_train, y_train, model):
    # Only works for exactly 2 features
    if X_train.shape[1] != 2:
        return  # silently skip (no popup spam)

    # Clear previous plot
    for widget in plot_frame.winfo_children():
        widget.destroy()

    X = np.array(X_train)
    y = np.array(y_train)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k")

    ax.set_title("Classification Decision Boundary")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)


'''
def plot_confusion_matrix_tkinter(y_test, y_pred):
    # Clear previous plot
    for widget in plot_frame.winfo_children():
        widget.destroy()

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)

    cm = confusion_matrix(y_test, y_pred)
    ax.imshow(cm)

    ax.set_title("Logistic Regression - Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    # Add values inside cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center")

    fig.tight_layout()

    # Embed plot into Tkinter
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
'''  
def plot_clusters(X, labels):
    if X.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='rainbow')
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title("Clustering Result")
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    else:
        messagebox.showinfo("Info", "Clustering plot requires at least 2 features.")


# ---------------- GUI LAYOUT ---------------- #

frame1 = tk.LabelFrame(root, text="Step 1: Upload Dataset")
frame1.pack(fill='x', padx=10, pady=5)
bot_upload = tk.Button(frame1, text="Upload CSV/Excel", command=upload_file)
bot_upload.pack(side="left", padx=10, pady=5)

frame2 = tk.LabelFrame(root, text="Step 2: Select Task and Algorithm")
frame2.pack(fill='x', padx=10, pady=5)

task_var = tk.StringVar()
algorithm_var = tk.StringVar()

tk.Label(frame2, text="Task:").pack(side="left", padx=5)
task_menu = ttk.Combobox(frame2, textvariable=task_var, values=["Regression", "Classification", "Clustering"], state="readonly")
task_menu.pack(side="left", padx=5)
task_menu.current(0)

tk.Label(frame2, text="Algorithm:").pack(side="left", padx=5)
algo_menu = ttk.Combobox(frame2, textvariable=algorithm_var, values=[], state="readonly")
algo_menu.pack(side='left', padx=5)

def update_algorithm(event):
    task = task_var.get()
    if task == "Regression":
        algo_menu['values'] = ["Linear Regression"]
    elif task == "Classification":
        algo_menu['values'] = ["Logistic Regression", "Decision Tree", "Random Forest", "Naive Bayes", "KNN", "SVM"]
    elif task == "Clustering":
        algo_menu['values'] = ["KMeans"]
    algo_menu.current(0)

task_menu.bind("<<ComboboxSelected>>", update_algorithm)

frame3 = tk.LabelFrame(root, text="Step 3: Specify Target Column (not needed for Clustering)")
frame3.pack(fill="x", padx=10, pady=5)
tk.Label(frame3, text="Target Column:").pack(side="left", padx=5)
target_column_var = tk.StringVar()
target_column_menu = ttk.Combobox(frame3,textvariable = target_column_var,state = "readonly")
target_column_menu.pack(side="left", padx=5)

frame4 = tk.LabelFrame(root, text="Step 4: Train Model")
frame4.pack(fill='x', padx=10, pady=5)
train_button = tk.Button(frame4, text="Train Model", command=train_model)
train_button.pack(fill='x', padx=10, pady=5)

# --- Combined Layout: Frame5 + Frame6 (left) and Plot Frame (right) --- #

# Create parent container for left (data/results) and right (plots)
result_display_frame = tk.Frame(root)
result_display_frame.pack(fill="both", expand=True, padx=10, pady=5)

# LEFT SIDE CONTAINER (Data Preview + Results)
left_frame = tk.Frame(result_display_frame)
left_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)

# Frame 5 : Data Preview
frame5 = tk.LabelFrame(left_frame, text="Data Preview")
frame5.pack(fill="both", expand=True, padx=5, pady=5)

text_preview = tk.Text(frame5, height=10)
text_preview.pack(fill="both", expand=True)

# Frame 6 : Result and Metrics
frame6 = tk.LabelFrame(left_frame, text="Result and Metrics")
frame6.pack(fill="both", expand=True, padx=5, pady=5)

result_text = tk.Text(frame6, height=10)
result_text.pack(fill="both", expand=True)

# RIGHT SIDE CONTAINER (Visualization)
plot_frame = tk.LabelFrame(result_display_frame, text="Visualization")
plot_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

root.mainloop()