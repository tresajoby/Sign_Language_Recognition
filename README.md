# Real-Time Sign Language Recognition using Python

![Sign Language Demo](https://i.imgur.com/your-demo-image.gif) <!-- Optional: Replace with a GIF of your app working -->

## 📖 Overview

This project is a real-time American Sign Language (ASL) recognizer built with Python. The application uses a computer's webcam to detect hand gestures and translates them into letters and numbers on the screen instantly.

The core of this project lies in using the **MediaPipe** library to extract hand landmark coordinates and a trained **Scikit-learn** model to classify these landmarks into their corresponding signs. This provides a fast, efficient, and AI-powered method for sign language detection.

## ✨ Features

-   **Real-Time Detection:** Recognizes hand signs instantly from a live webcam feed.
-   **Comprehensive Alphabet & Numbers:** The included model is trained to recognize a wide range of ASL signs, including letters (A-Z) and numbers (0-9).
-   **AI-Powered:** Utilizes a trained `RandomForestClassifier` for accurate and robust predictions based on hand landmark data.
-   **Interactive UI:** A simple desktop interface built with `OpenCV` displays the webcam feed, the detected hand landmarks, and the predicted character.
-   **Extensible:** The project is structured to be easily extensible. You can train the model on new signs or even full words to expand its vocabulary.

## 🛠️ Technology Stack

-   **Python 3.9+**
-   **OpenCV:** For capturing and displaying the webcam feed.
-   **MediaPipe:** For high-fidelity hand tracking and landmark extraction.
-   **Scikit-learn:** For loading and using the trained classification model (`model.p`).
-   **Tkinter (built-in):** Used for the graphical user interface.

## 🚀 Getting Started

Follow these instructions to get the project up and running on your local machine.

### 1. Prerequisites

You need to have `conda` (or `miniconda`) installed to manage the Python environment.

### 2. Installation & Setup

It is highly recommended to create a dedicated Conda environment to avoid conflicts with other Python projects.


