# PM 2.5 Prediction (regression)


## Task
training data: 同一年某地區的資料當中取樣出數天，以連續的24小時為一組數據，第k~k+7小時的觀測數據當作 train_X，第 k+8 小時的 PM2.5 當作 train_y。
testing data : 同一年某地區的資料當中取樣出數天，以連續的9小時為一組數據，前8小時的觀測數據當作 test_X，請預測第九小時的 PM2.5 當作 train_y。
一共預測 90 筆第九小時的 PM2.5。

## Data
Data含有 15 項數據可作為特徵: 
`AMB_TEMP, CO, NO, NO2, NOx, O3, PM10, WS_HR, RAINFALL, RH, SO2, WD_HR, WIND_DIREC, WIND_SPEED, PM2.5`

# Math Problem

## Topics
1. Closed-Form Linear Regression Solution
3. Logistic Sigmoid Function and Hyperbolic Tangent Function
4. Noise and Regulation
5. Logistic Regression
