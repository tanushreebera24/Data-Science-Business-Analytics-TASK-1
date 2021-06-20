# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 12:32:02 2021

@author: SOLUTION
"""

{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "db826834",
   "metadata": {},
   "source": [
    "# The Sparks Foundation \n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0e718d73",
   "metadata": {},
   "source": [
    "# Data Science and Business Analytics Internship Task 1 "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2debab73",
   "metadata": {},
   "source": [
    "# Task by: Ashish Handa"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "24a051c4",
   "metadata": {},
   "source": [
    "##  Prediction using Supervised ML\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "feff6ce8",
   "metadata": {},
   "source": [
    "### Step 1: Understand the Problem\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "25792efd",
   "metadata": {},
   "source": [
    "Predict the percentage of a student based on the no. of study hours."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0c0b2d3e",
   "metadata": {},
   "source": [
    "### Step 2: Find the source of data and load it.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cefc49be",
   "metadata": {},
   "source": [
    "Dataset is available at: https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "75cd0150",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "b860f98f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours</th>\n",
       "      <th>Scores</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2.5</td>\n",
       "      <td>21</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5.1</td>\n",
       "      <td>47</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3.2</td>\n",
       "      <td>27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8.5</td>\n",
       "      <td>75</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.5</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Hours  Scores\n",
       "0    2.5      21\n",
       "1    5.1      47\n",
       "2    3.2      27\n",
       "3    8.5      75\n",
       "4    3.5      30"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv(\"C:\\\\Users\\\\Rahul Handa\\\\Desktop\\Data.csv\")\n",
    "\n",
    "\n",
    "\n",
    "# Printing the first 5 rows of the dataset\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "23a4d492",
   "metadata": {},
   "source": [
    "### Step 3: Explore the data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "9aa89bf3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 25 entries, 0 to 24\n",
      "Data columns (total 2 columns):\n",
      " #   Column  Non-Null Count  Dtype  \n",
      "---  ------  --------------  -----  \n",
      " 0   Hours   25 non-null     float64\n",
      " 1   Scores  25 non-null     int64  \n",
      "dtypes: float64(1), int64(1)\n",
      "memory usage: 464.0 bytes\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "cfe7235b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours</th>\n",
       "      <th>Scores</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>25.000000</td>\n",
       "      <td>25.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>5.012000</td>\n",
       "      <td>51.480000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>2.525094</td>\n",
       "      <td>25.286887</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.100000</td>\n",
       "      <td>17.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>2.700000</td>\n",
       "      <td>30.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>4.800000</td>\n",
       "      <td>47.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>7.400000</td>\n",
       "      <td>75.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>9.200000</td>\n",
       "      <td>95.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           Hours     Scores\n",
       "count  25.000000  25.000000\n",
       "mean    5.012000  51.480000\n",
       "std     2.525094  25.286887\n",
       "min     1.100000  17.000000\n",
       "25%     2.700000  30.000000\n",
       "50%     4.800000  47.000000\n",
       "75%     7.400000  75.000000\n",
       "max     9.200000  95.000000"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6dad4f4c",
   "metadata": {},
   "source": [
    "### Step 4: Visualize the data.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "dade0a47",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "6b987766",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(df.Hours, df.Scores)\n",
    "plt.title(\"Hours Vs Scores\")\n",
    "plt.xlabel(\"------ Hours ----->\")\n",
    "plt.ylabel(\"------ Scores ----->\")\n",
    "plt.grid()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d2a4948e",
   "metadata": {},
   "source": [
    "### Step 5: Model Selection and model training.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "74672dd3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Taking input column in x\n",
    "x = df['Hours']\n",
    "\n",
    "# Taking output/target column in y\n",
    "y = df['Scores']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "c0897540",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    2.5\n",
       "1    5.1\n",
       "2    3.2\n",
       "3    8.5\n",
       "4    3.5\n",
       "Name: Hours, dtype: float64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "d97837ba",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    21\n",
       "1    47\n",
       "2    27\n",
       "3    75\n",
       "4    30\n",
       "Name: Scores, dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "8d7ce27b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Splitting the data into train and test for model building\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "20b88c73",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(18, 7)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(x_train), len(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "dc07c3d7",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "\n",
    "\n",
    "model = LinearRegression()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "b0d78c5f",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train = x_train.values.reshape(-1, 1)\n",
    "y_train = y_train.values.reshape(-1, 1)\n",
    "\n",
    "x_test = x_test.values.reshape(-1, 1)\n",
    "y_test = y_test.values.reshape(-1, 1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "f77570d6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(x_train, y_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "794102c4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plotting the mx + c line over test data\n",
    "\n",
    "m = model.coef_[0]\n",
    "c = model.intercept_\n",
    "\n",
    "\n",
    "x_line = np.arange(0, 10, 0.1)\n",
    "\n",
    "y_line = m * x_line + c\n",
    "\n",
    "plt.plot(x_line, y_line, \"r\")\n",
    "\n",
    "plt.scatter(x_test, y_test)\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2a9379b6",
   "metadata": {},
   "source": [
    "### Step 6: Model Testing."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "e50e20dc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9367661043365055"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.score(x_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "b606070c",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = model.predict(x_test)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "77cbd66f",
   "metadata": {},
   "source": [
    "### What will be predicted score if a student studies for 9.25 hrs/ day?\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "58f59532",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "93.89272889341655"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "user_input = np.array(9.25).reshape(1, -1)\n",
    "\n",
    "\n",
    "answer = model.predict(user_input)\n",
    "\n",
    "answer[0][0]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2e462d13",
   "metadata": {},
   "source": [
    "### Predicted score if a student studies for 9.25 hrs/ day: 93.89272889341655\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
