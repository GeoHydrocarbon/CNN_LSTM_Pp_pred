# Formation pore fluid pressure (FPFP) prediction based on hybrid CNN-LSTM deep learning method

Chong Yang<sup>a</sup>, Junlin Chen<sup>a</sup>, Mingliang Liu<sup>b</sup>, Ge Jin<sup>c</sup>, Xiaowen Guo<sup>a, \*</sup>

<sup>a</sup> Key Laboratory of Tectonics and Petroleum Resources Ministry of Education, China University of Geosciences, Wuhan, 430074, China, ORCID(s): 0009-0007-2530-4361

<sup>b</sup> School of Future Technology, Shandong University, Jinan, 250100, China

<sup>c</sup> Colorado School of Mines, 1500 Illinois Street, Golden, CO 80401, USA

\* Corresponding author. China University of Geosciences, China.

E-mail addresses: cuggxw@163.com.

ARTICLE INFO

Keywords:

machine learning

well logging data

formation pore fluid pressure

CNN-LSTM

Authorship contribution statement

Chong Yang: Writing – original draft, Visualization, Validation, Investigation, Data Curation. Junlin Chen: Writing – review & editing, Validation, Resources, Methodology, Supervision. Ge Jin: Writing – review & editing, Validation, Methodology. Xiaowen Guo: Writing – review & editing, Validation, Methodology.

# Abstract

Accurate pressure prediction is of vital importance in oil and gas exploration and development. Traditional methods like Eaton’ and Bowers’ for predicting formation pore fluid pressure (FPFP) are limited when forming particular structures and lithology. This study proposes an FPFP prediction method based on the deep learning model CNN-LSTM. And its performance in pressure prediction was compared with that of traditional machine learning (ML) models (including Decision Tree (DT), Random Forest (RF), XGBoost, CatBoost, MLP, SVR). This study collected well log data (sonic, density, gamma ray, neutron porosity and deep resistivity) from 6 wells, each with 1-3 measured pressure points. Well 1 is used as training set, and well 2-6 are used as test set. These measured pressure data and test set are utilized to evaluate the model training results and validate the prediction accuracy. According to the experimental results, the performance of the CNN-LSTM model in the 5-fold cross-validation is highly consistent. The R² values of each fold are close to 0.998, which are higher than traditional ML method whose max R² value is lower than 0.980. The MSE and MAE of CNN-LSTM are lowest. The average absolute error of predicted pressure and measured pressure of 5 wells is 5.760%, which is much lower than traditional ML method whose lowest error is 8.694% (SVR), max error is 24.019% (DT). This suggests the CNN-LSTM model can better capture the complex nonlinear relationships and temporal dependencies in the data compared with traditional machine learning models, providing more accurate and consistent pressure prediction results in the prediction of multiple Wells, demonstrating excellent learning and generalization ability.

# 1\. Introduction

Accurate prediction of formation pore fluid pressure (FPFP) is a cornerstone of safe, efficient, and economical hydrocarbon exploration and development. Pore fluid pressure, defined as the pressure exerted by fluids contained within the interconnected pore spaces of subsurface rock formations(O’connor et al., 2011; Zhang, 2011), is a critical geomechanically parameter. Its accurate estimation is paramount for well planning, casing design, drilling mud weight optimization, mitigating drilling hazards (such as kicks, blowouts, and wellbore instability), and understanding basin evolution and trap integrity(de Souza et al., 2021; Jafarizadeh et al., 2022). Deviations from the normal hydrostatic pressure gradient - specifically, overpressure (pressure significantly exceeding hydrostatic) – are common phenomena in sedimentary basins worldwide (Shi et al., 2013; Zhao et al., 2018). These overpressured zones pose significant risks and complexities during drilling operations(Banerjee and Chatterjee, 2021; McConnell et al., 2012). The genesis of overpressure is multifaceted, encompassing disequilibrium compaction(Hua et al., 2021; Zhang et al., 2024; Zhao et al., 2018), tectonic stresses (compression, faulting)(Zhang et al., 2024), fluid expansion (hydrocarbon generation, aquathermal effects)(Liu et al., 2021), osmosis(Mark J. Osborne and Richard E. Swar, 1997), and mineral transformations(Henning Dypvik, 1983).

Traditional approaches to FPFP prediction primarily rely on empirical or semi-empirical relationships derived from geophysical well logs (e.g., sonic velocity, resistivity) and seismic velocity data.(Zhang et al., 2022) Methods like Eaton’s (Eaton, 1975) and Bowers’ (Bowers, 1995) are widely employed, relating observed parameters (like sonic travel time or seismic interval velocity) to effective stress and subsequently calculating pore pressure using Terzaghi’s principle (Pore Pressure = Overburden Stress - Effective Stress) (Lade and De Boer, 1997). A key constraint of these conventional techniques, however, is their reliance on pressure calibration measurements for accurate results—these direct pressure data (e.g., drill stem test (DST), modular dynamic tester (MDT), and repeat formation test (RFT)) are not only expensive to acquire but also often sparse and obtained only post-drilling (Zhang, 2011). In contrast, the data-driven method proposed in this study only requires pressure measurements from a single well for model training. After sufficient training and validation on wells with available pressure measurement data, the model can accurately predict FPFP for other wells. This advantage significantly reduces reliance on costly pressure calibration tests and expands the applicability of pressure prediction in areas with limited direct pressure data, thereby highlighting the practical value of the proposed method.

The advent of machine learning (ML) and deep learning (DL) offers a paradigm shift, promising to overcome the constraints of conventional models. ML algorithms excel at identifying complex, non-linear patterns within large, multi-faceted datasets without requiring explicit physical equations (Lary et al., 2016; Sarker, 2021). Recent research demonstrates the superior predictive capabilities of ML models (e.g., MLP, SVM, RF, DT, XGboost and Catboost) compared to parametric methods, achieving significantly lower prediction errors (Yu et al., 2020). Deep learning, a subset of ML utilizing hierarchical neural networks, provides even greater potential. Specifically, Convolutional Neural Networks (CNNs) have proven exceptionally adept at extracting salient spatial features and patterns from structured grid data, such as seismic attribute volumes, well log image representations, and geological maps (An et al., 2023; Hussain et al., 2025). Their ability to automatically learn relevant spatial hierarchies makes them ideal for capturing the geological context influencing FPFP distribution. Concurrently, Long Short-Term Memory (LSTM) networks, a specialized type of Recurrent Neural Network (RNN), address a critical weakness in modeling temporal sequences: the long-term dependency problem. LSTMs utilize a gated cell structure (input, forget, output gates) to selectively retain, update, and output information, effectively capturing temporal dynamics and evolutions over extended sequences (Greff et al., 2017; Sahoo et al., 2019; Zheng and Huang, 2017). Recognizing the complementary strengths of these architectures-CNN for spatial feature extraction and LSTM for temporal sequence modeling—hybrid CNN-LSTM models have emerged as a powerful framework for spatiotemporal prediction tasks. Studies consistently report that hybrid CNN-LSTM models outperform standalone CNN, LSTM, or traditional ML models in accuracy and robustness (Gilik et al., 2022; Zhang et al., 2023).

In this paper, we compare the FPFP prediction resolution of ML method with DL method and present a novel CNN-LSTM deep learning framework specifically designed for high-resolution, reliable prediction of FPFP based on normal well logs (e.g., AC, DEN, GR, LLD, CNL). The core objective is to demonstrate the model’s robustness and superior predictive accuracy, particularly in challenging geological environments characterized by structural complexity, lithological heterogeneity, and multi-mechanism overpressure generation. By achieving these goals, this research seeks to provide a significant advancement in FPFP prediction methodology, enhancing the safety and efficiency of subsurface resource exploration and development.

# 2\. Methodology

In this paper, the CNN-LSTM for FPFP prediction is presented. Fig. 1 illustrates a specific implementation process of the proposed method of this paper.

Fig. 1 Process of ML and CNN-LSTM methods for FPFP prediction.

## 2.1. Dataset preparing

Machine learning (ML) and deep learning (DL) techniques typically require large volumes of data to effectively predict pore pressure in subsurface formations. However, directly measuring FPFP from wells (via test like drill stem test (DST), modular dynamic tester (MDT), and repeat formation test (RFT)) is prohibitively expensive and operationally limited, resulting in extremely sparse datasets. Consequently, due to this scarcity of direct pressure measurements for training, researchers have explored the approach of using predicted effective stress derived from well logs (using traditional methods like Eaton, Bowers, or Equivalent depth) as the target output or training labels for ML/DL models instead.

In this paper, six wells’ logging data were prepared for the experiment. And five type of logging curves including sonic (AC), density (DEN), gamma ray (GR), neutron porosity (CN) and deep resistivity (LLD) are selected for training (Fig. 2). The well logging data of Well 1 was used generate training labels through Eaton’s method (Eaton, 1975), and two measured pressure points were used to adjust the generated labels to ensure the labels are reliable. The FPFP of the other five Wells was predicted by the ML and CNN-LSTM models trained based on the well logging data from Well 1. Each well has 1 ~ 3 measured pressure points to evaluate the generalization ability of all models.

Eaton’s equation is defined as:

Where Pp is the predicted formation pressure, MPa; σ<sub>v</sub> is the overburden stress, MPa; Pw is the normal hydrostatic pressure, MPa; ∆t is the normal compaction sonic transit time, μs/ft; ∆t<sub>i</sub> is the measured shale sonic transit time, μs/ft; and n is the Eaton exponent, dimensionless.

Specifically, the normal compaction trend of acoustic travel time in the study area was first determined based on the shallow normal pressure interval of Well 1. Then, the formation pore pressure was calculated via the classical Eaton equation. Subsequently, the Eaton exponent was modified using a total of two measured pressure data points obtained from DST in this well—these points cover shallow normal pressure interval (Well 1:2973.9 m) and deep overpressure intervals (Well 1:3249.65 m）. This distribution ensures the modified Eaton exponent can reflect both normal and overpressure systems, thereby improving the geological rationality of the initial pressure labels which were further calibrated to ensure the reliability and validity of the training dataset. Finally, the logging data of this well were merged to construct the training samples, based on which machine learning (ML) and CNN-LSTM hybrid models were trained to predict the formation pore fluid pressure (FPFP) of the remaining 5 wells.

Fig. 2 Well logging curves used to build the training dataset for Well 1.

## 2.2. Data preprocessing

### 2.2.1. Data ****standardization****

To ensure consistent scales across all input features and promote stable model convergence, each well log variable was standardized via the StandardScaler method from scikit-learn, which applies Z-score standardization (centering to zero mean and scaling to unit variance). This is critical for CNN-LSTM model, as unbalanced feature scales may cause unstable gradients and hinder convergence (Sun et al., 2024).

The transformation is defined as:

Where 𝑥 is the original feature value, μ is the mean of the feature in the training set, σ is the standard deviation of the feature in the training set, ​ is the normalized output.

This standardization preserves the relative relationships between data points while transforming features to a common scale (mean=0, std=1), essential for gradient-based optimization and distance-sensitive models. To prevent data leakage, the standardization parameters (mean and standard deviation) were derived solely from the training set and then applied uniformly to the validation and test sets.

### 2.2.2. Correlation analysis of features

To assess the linear relationships among input variables and between input features and the target pore pressure, a **correlation analysis** was conducted using the **Pearson correlation coefficient**. This statistical measure quantifies the strength and direction of linear dependence between two continuous variables, which is particularly valuable in the context of machine learning model development.

The Pearson correlation coefficient _r_ between two variables _x_ and _y_ is defined as:

where and are the mean values of _x_ and _y_, respectively. The coefficient ranges from -1 to +1, with +1 indicating perfect positive linear correlation, -1 indicating perfect negative correlation, and 0 indicating no linear correlation.

### 2.2.3. Model evaluation metrics: MSE, MAE and 𝑅<sup>2</sup> Score

To quantitatively assess the predictive performance of the machine learning models, three standard regression evaluation metrics were employed: Mean Squared Error (MSE), Mean Absolute Error (MAE), and the Coefficient of Determination (R²). These metrics provide complementary insights into both the absolute prediction accuracy and the proportion of variance in the target variable that is captured by the model (Yu et al., 2020).

**MSE** measures the average squared difference between the predicted values and the actual values _y_, and is defined as:

Lower MSE values indicate better model performance, as they reflect smaller average prediction errors. Due to the squaring operation, MSE is particularly sensitive to large deviations, which helps identify models that produce occasional but critical outliers—an important consideration in pressure prediction tasks.

**MAE**, on the other hand, measures the average absolute difference between the predicted and actual values, and is defined as:

MAE provides a more straightforward interpretation, as it represents the average magnitude of the prediction errors without emphasizing large deviations as much as MSE. Lower MAE values indicate better model accuracy.

**The** R<sup>2</sup> **score** (coefficient of determination) evaluates the proportion of variance in the actual data that is explained by the predicted values. It is calculated as:

where is the mean of the observed values. An R<sup>2</sup> value close to 1 indicates that the model captures most of the variability in the data, whereas values near 0 suggest poor predictive power. In the context of well log-based pore pressure modeling, a high R<sup>2</sup> score reflects that the model successfully replicates depth-wise pressure trends derived from geophysical data.

By combining MSE, MAE, and R², a more comprehensive understanding of the model’s performance is obtained, covering both error sensitivity (MSE), overall prediction accuracy (MAE), and the variance explained (R²).

## 2.3. Hybrid Deep Learning Architecture: CNN–LSTM for FPFP

### 2.3.1. Theoretical Background

The prediction of FPFP from well logs involves learning both local spatial features and sequential temporal patterns inherent in depth-aligned geophysical measurements. To capture these aspects effectively, this study adopts a hybrid deep learning architecture that combines **Convolutional Neural Networks (CNN)** and **Long Short-Term Memory (LSTM)** networks**.**

**CNNs** are well-suited for extracting localized and hierarchical spatial features from structured input data, such as sliding windows of multivariate well logs. In this context, CNN layers learn spatial correlations across log types and depth positions, enabling the model to detect high-resolution geological patterns (An et al., 2023; Hussain et al., 2025).

**LSTM networks**, a specialized form of recurrent neural networks (RNNs), are designed to model long-range dependencies in sequential data (Greff et al., 2017; Sahoo et al., 2019). Their internal gating mechanism (input, forget, and output gates) allows them to retain relevant historical information over extended depth intervals (Zheng and Huang, 2017). This capability is critical in subsurface modeling, where pore pressure behavior often evolves gradually with depth and reflects cumulative geological processes.

By combining CNN and LSTM, the model leverages the feature extraction strength of CNN and the temporal memory capabilities of LSTM, making it well-suited for learning from time-depth structured well log data in a data-driven manner.

### 2.3.2. Model Implementation

The hybrid CNN–LSTM model was implemented using PyTorch, and its architecture consists of the following components: To extract log curve features at different time scales, the model designs a multi-branch one-dimensional convolution module (InceptionConv1D). This module consists of three parallel branches, which respectively use convolution operations with kernel sizes of 3, 5, and 7 to extract short-term, medium-term, and long-term feature representations. The outputs of all branches are concatenated in the channel dimension to form a unified feature representation, enabling multi-scale feature fusion in a single module. Based on the multi-scale features extracted by CNN, the model uses a double-layer LSTM structure to model the time series information. The LSTM unit can effectively capture long-term dependencies and enhance the model’s understanding of the implicit time series structure in geological logging data. The hidden state of the last time step in the LSTM output sequence is input to the fully connected layer, and the predicted value is ultimately output. (Fig. 3)

Divide the training set into 5 subsets. In each round, select one of them as the validation set and the rest as the training set. Optimization is carried out using the MSE and the Adam optimizer. Introduce the Early Stopping strategy to prevent overfitting and terminate the training prematurely when the validation loss no longer decreases for several consecutive rounds. After each fold of training is completed, evaluate its MAE and R² on the validation set, and draw the training and validation loss curves to observe the convergence of the model. When training the final model, the entire training set is used for model fitting without partitioning the validation set or using Early Stopping. The number of training rounds is reduced to half (50 times), and the loss curve during the training process is recorded simultaneously.

In the end, depth‑aligned line plots of predicted versus actual pore‑pressure were produced to assess agreement across the entire well profile.

Fig. 3. Architecture of the hybrid CNN-LSTM model for FPFP prediction.

## 2.4. Machine learning algorithms for FPFP

### 2.4.1. Support vector Machine

Support Vector Machine is a supervised learning algorithm initially designed for classification but effectively extended to regression problems, known as Support Vector Regression (SVR). It operates by identifying an optimal hyperplane in a high-dimensional feature space to maximize the margin between support vectors, thereby minimizing the generalization error through structural risk minimization. For regression tasks, SVR employs an ε-insensitive loss function to tolerate small deviations, making it robust to noise and outliers (Frénay and Verleysen, 2011). This kernel-based method can handle non-linear relationships via kernel tricks (e.g., radial basis function), which implicitly map data into higher dimensions without explicit transformation (Greff et al., 2017). SVM is particularly suitable for high-dimensional datasets and scenarios requiring strong generalization, such as financial risk prediction or engineering simulations, though it may be computationally intensive for large-scale data.

### 2.4.2. Multilayer Perceptron (MLP)

Multilayer Perceptron is a class of feedforward artificial neural network characterized by multiple layers of interconnected neurons, including an input layer, one or more hidden layers with activation functions (e.g., ReLU), and an output layer that produces continuous predictions for regression. It learns non-linear mappings through backpropagation, which adjusts weights to minimize a loss function (e.g., MSE) via gradient descent. MLPs can approximate complex functions and capture intricate patterns in high-dimensional data, but they require careful initialization and regularization to avoid overfitting and ensure convergence. In regression applications, such as water quality prediction or structural deformation modeling, MLPs excel with sufficient data and computational resources (Kim, 2016).

### 2.4.3. Decision tree

Decision Tree is a non-parametric supervised learning algorithm that models decisions and their potential consequences using a tree-like structure, with internal nodes representing feature-based splits and leaf nodes providing predicted continuous values for regression tasks. It recursively partitions the input space into regions based on impurity minimization criteria (e.g., mean squared error reduction), allowing for intuitive interpretation and handling of non-linear relationships. While simple and efficient for real-time decision-making, DTs are prone to overfitting, especially with deep trees (Nissa et al., 2024).

### 2.4.4. Random Forest

Random Forest is an ensemble learning method that constructs multiple decision trees during training and aggregates their predictions through averaging (for regression) or voting (for classification) to enhance accuracy and reduce overfitting. Each tree is built using a random subset of the training data (bootstrap sampling) and a random subset of features at each split, introducing diversity among the trees. This randomness decorrelates individual tree errors, leading to improved stability and predictive performance. In regression, RF minimizes variance by combining numerous weak learners, making it effective for handling noisy or missing data. Applications include ecological modeling and supply chain risk assessment, where it provides interpretable feature importance scores, but it may require careful parameter tuning to avoid bias in imbalanced datasets (Stumpf and Kerle, 2011).

### ****2.4.5.**** Advanced Gradient Boosting Algorithms

XGBoost and CatBoost are both advanced gradient boosting algorithms optimized for efficiency and handling specific data types. XGBoost is a scalable and high-performance implementation of gradient boosting designed for regression and classification tasks. It builds an additive model by sequentially fitting decision trees to the residuals of previous predictions, minimizing a differentiable loss function (e.g., squared error for regression). Key innovations include a sparsity-aware algorithm for handling missing values, weighted quantile sketches for tree learning, and L1/L2 regularization to control model complexity and prevent overfitting. With parallelized tree construction and resource-efficient techniques like cache-aware access, XGBoost excels in large-scale datasets and often achieves state-of-the-art results in challenges such as bioinformatics and fraud detection (Hasan et al., 2021).

CatBoost employs an asymmetric tree-growing strategy and ordered boosting to reduce prediction shift and overfitting, while incorporating permutations of categorical variables to capture interactions. For regression tasks, CatBoost optimizes the loss function through gradient-based steps with built-in numerical stability, such as automatic handling of feature combinations. It often outperforming other tree-based methods in terms of accuracy and robustness, especially when dealing with imbalanced or high-cardinality features (Maritan, 2025)..

# 3\. Results and discussion

Fig. 4 shows the correlation matrix heatmaps of training data from Well 1. There is a very strong positive correlation (0.92) between Depth and FPFP, indicating that depth is a key feature for predicting labels. The AC and DEN have a strong positive correlation with FPFP (0.72 and 0.61), respectively. These features have a certain linear relationship with depth and can provide useful information for the model. There is a negative correlation (-0.34) between GR and FPFP, but it may still contribute to prediction. The positive correlation between resistivity (LLD) and FPFP is 0.43, suggesting that it also plays a certain role in prediction. The correlation between neutron porosity (CN) and FPFP is relatively weak (0.16), which may have a limited contribution to the prediction of labels.

Fig. 4. Pearson correlation heatmaps of the training data.

## 3.1. Model evaluation

### 3.1.1. cross-validated R² score

To evaluate the performance and stability of different models in the stress prediction task, we adopted 5-fold cross-validation. This method divides the training dataset into five subsets. Four subsets are used for training each time, and the remaining one subset is used to validate the model’s performance. This process is repeated five times, with the validation set being changed each time. In each fold, we trained different models and recorded the prediction error of each fold. This method can help us assess the stability of the model, reduce the risk of overfitting, and ensure that the model performs consistently on different training and validation data.

The results show that the CNN-LSTM model consistently outperforms other models in all folds, achieving the highest R² values (Fig. 5). Specifically, the R² values of CNN-LSTM in the 5-fold cross-validation are 0.9983, 0.9985, 0.9981, 0.9984, and 0.9980 respectively (Table 1), demonstrating extremely high prediction accuracy and stability. In contrast, the R² values of traditional machine learning models fluctuate across different folds. Although RF and XGBoost perform well in some folds (with the maximum R² values of 0.9721 and 0.9673 respectively), they never surpass the CNN-LSTM model.

Specifically, the DT performed poorly in all folds, with its R² value being below 0.94 in each fold, indicating its relatively weak prediction accuracy and stability. The SVR also performed relatively weakly in the 5-fold cross-validation, with its R² value fluctuating significantly across different folds. Through these results, it is clearly seen that CNN-LSTM not only performed stably in each fold but also outperformed other models in all folds, demonstrating its strong performance and adaptability in the pressure prediction task.

|     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5-fold<br><br>cross-validated | DT  | RF  | XGBoost | CatBoost | MLP | SVR | CNN-LSTM |
| Fold1 | 0.9176 | 0.9600 | 0.9533 | 0.9597 | 0.9485 | 0.9385 | 0.9983 |
| Fold2 | 0.9228 | 0.9571 | 0.9545 | 0.9584 | 0.9499 | 0.9302 | 0.9985 |
| Fold3 | 0.9156 | 0.9685 | 0.9629 | 0.9665 | 0.9579 | 0.9417 | 0.9981 |
| Fold4 | 0.9390 | 0.9705 | 0.9673 | 0.9684 | 0.9555 | 0.9402 | 0.9984 |
| Fold5 | 0.9411 | 0.9721 | 0.9651 | 0.9696 | 0.9605 | 0.9456 | 0.9980 |

Table 1. 5-fold cross-validated R2 score of ML and CNN-LSTM models.

Fig. 5 Box plot of 5-fold cross-validated R2 score of the ML and CNN-LSTM models.

### 3.1.2. MSE and MAE

MSE and MAE were also used as the evaluation indicators. From the results, it is evident that the CNN-LSTM model outperforms traditional machine learning models significantly in the 5-fold cross-validation. The MSE and MAE values for each fold are relatively low. Specifically, the MSE values of CNN-LSTM range from 0.22 to 0.42 across Fold1 to Fold5, and the MAE values range from 0.35 to 0.48 (Fig. 6 and Table 2). These results indicate that CNN-LSTM can effectively capture the complex nonlinear relationships in the pressure prediction task and maintain high prediction accuracy and stability across multiple different data subsets. In contrast, the MSE and MAE values of other models are significantly higher, especially for DT (MSE = 10.48, MAE = 1.90) and SVR (MSE = 8.05, MAE = 1.84), suggesting that they perform poorly in terms of prediction accuracy and stability. These results indicate that CNN-LSTM can not only provide highly accurate predictions, but also maintain a low prediction error in different datasets and cycles, demonstrating its robustness and reliability in handling such complex tasks.

Fig. 6. Comparison of MSE and MAE between the ML models and the CNN-LSTM model (5 folds and full test).

### 3.1.3. Comparison between the predicted and the actual pressure

The predicted values of the training models of the six ML algorithms and the CNN-LSTM training model were all consistent with the measured pressure points (Fig. 7 and Fig. 8). Fig. 9 and Fig. 10 show the comparison between the predicted values of different models and the actual pressure values. It can be clearly seen that the prediction curve of the CNN-LSTM model almost completely coincides with the real pressure curve, while the prediction results of other traditional models have certain deviations, especially in deeper strata, where this deviation is more obvious. The CNN-LSTM model can fit deep-level pressure data more accurately, and the error distribution is relatively uniform, with no significant fluctuations or outliers.

Fig. 7. Comparison of predicted pressure profile, actual pressure profile and measured pressure points of ML models.

Fig. 8. Comparison of predicted pressure profile, actual pressure profile and measured pressure points of the CNN-LSTM (5 folds and full test) model.

Fig. 9. Comparison of predicted pressure and actual pressure of the ML models.

Fig. 10. Comparison of predicted pressure and actual pressure of the CNN-LSTM (5 folds and full test) model.

Fig. 11 shows the residual distribution of different ML models, and Fig. 12 shows the residual distribution of the CNN-LSTM model. Residual is the difference between the predicted value of the model and the true value. A residual close to zero indicates that the model’s prediction is more accurate. It can be seen from the residual distribution graph that the residual distribution of CNN-LSTM is very concentrated, and most of the residual values are close to zero, indicating that its prediction error is very small. In contrast, the residual distribution of other machine learning models is relatively scattered, especially for SVR and DT models, which have larger residuals, indicating that the prediction errors of these models are more significant.

Fig. 11 Comparison of residual of the ML model.

Fig. 12 Comparison of residual of the CNN-LSTM (5 folds and full test) model.

## 3.2. Models’ performance in FPFP prediction

In this study, the trained models were used to predict the pressure of other unknown pressure Wells, which contained 1 to 3 measured pressure points. The predicted pressure profile of each well predicted by different models can be seen from Fig. 15 to Fig. 24. To evaluate the prediction performance of the models, we calculated the absolute errors between the predicted pressure and the measured pressure points, and visualized the result as an error heat map (Fig. 13) and a box plot (Fig. 14). Meanwhile, the percentage of error has also been statistically analyzed in the Table 3. Through these results, we can have a more comprehensive understanding of the predictive performance of each model in different Wells.

It can be seen from the heat map that the prediction error of the CNN-LSTM model in all Wells is small, showing good stability and accuracy. Especially in the prediction of Wells such as Well 2 (3042.4m) and Well 3 (3394.9m), the absolute errors of CNN-LSTM (Fold2) are 1.43 and 0.53 respectively. In contrast, traditional machine learning models (such as DT and RF) have relatively large prediction errors in some Wells. Especially in the prediction of Well 3 (3394.9m), the error of DT is as high as 30.77, and that of RF is 26.54. These relatively large errors reflect the instability of these models at specific data points.

Fig. 13. Heat map shows absolute error (unit: MPa) between measured pressure points and predicted pressure of ML models and CNN-LSTM (5 folds and full test) model.

From the box plot (Fig. 14), we can observe that the error distribution of the CNN-LSTM model is relatively compact, and the interquartile range is small, demonstrating a lower prediction error and better stability. In contrast, the error distribution of traditional machine learning models (such as DT, RF, and SVR) is relatively wide, and there are more outliers. Especially in DT and SVR models, the error fluctuates greatly, reflecting the unstable performance of these models at different levels.

Fig. 14. Box plot shows absolute error between measured pressure points and predicted pressure of ML models and CNN-LSTM (5 folds and full test) model.

In the table of error percentages, we can see that traditional machine learning models such as XGBoost and CatBoost have significant errors in the prediction of certain Wells. For example, when the measured pressure of Well 3 (at a depth of 3,394.9m) is 60.74 MPa, the prediction error of XGBoost is 15.099%, and that of CatBoost is 14.213%. These relatively high error values reflect that the prediction stability of these models is poor when dealing with some complex Wells. Geologically, this is because traditional ML models fail to capture the coupling effects of multiple overpressure mechanisms. For example, DT and SVR treat input features as independent variables, ignoring the synergistic impact of AC (reflecting compaction) and GR (reflecting organic matter content) on overpressure—whereas the CNN-LSTM’s multi-scale feature extraction and sequential modeling can resolve such geological couplings. In contrast, the CNN-LSTM model performed significantly better than other models in the prediction of all Wells. In all 5-fold cross-validations, the error of CNN-LSTM remains in a very low range, especially in the predictions of Fold 2 (with an error of 0.942%), where the error values are significantly lower than those of other machine learning models. Furthermore, the error of the Full Test trained with all the data in the end was 1.794%, further verifying the high accuracy of the CNN-LSTM model when dealing with the actual pressure prediction task. The average error of the CNN-LSTM model of Fold2 (5.760%) is significantly lower than that of other machine learning models. For instance, the lowest average error of ML models is SVR (8.694%). Others models are all over 10%, like the average error of DT is 24.019%, that of RF is 21.435%, while that of XGBoost is 18.414%. These results indicate that CNN-LSTM not only performs well in the prediction of individual Wells, but also shows very stable and accurate performance across the entire dataset.

This advantage can be attributed to the deep learning characteristics of the CNN-LSTM architecture, especially the synergy between CNN in local feature extraction and LSTM in capturing temporal dependencies. Traditional machine learning models, such as XGBoost and SVR, although performing well in some cases, usually struggle to handle complex nonlinear relationships and temporal characteristics in data. In contrast, CNN-LSTM can better capture these complex patterns through its multi-level network structure, thereby improving the prediction accuracy.

Fig. 15. FPFP profile of Well 2 predicted by the CNN-LSTM (5 folds and full test) model.

Fig. 16. FPFP profile of Well 2 predicted by the ML models.

Fig. 17. FPFP profile of Well 3 predicted by the CNN-LSTM (5 folds and full test) model.

Fig. 18. FPFP profile of Well 3 predicted by the ML models.

Fig. 19. FPFP profile of Well 4 predicted by the CNN-LSTM (5 folds and full test) model.

Fig. 20. FPFP profile of Well 4 predicted by the ML models.

Fig. 21. FPFP profile of Well 5 predicted by the CNN-LSTM (5 folds and full test) model.

Fig. 22. FPFP profile of Well 5 predicted by the ML models.

Fig. 23. FPFP profile of Well 6 predicted by the CNN-LSTM (5 folds and full test) model.

Fig. 24. FPFP profile of Well 6 predicted by the ML models.

# 4\. Conclusions

This study conducted pressure prediction by adopting multiple machine learning models and deep learning models (including DT, RF, XGBoost, CatBoost, MLP, SVR, and CNN-LSTM), with a focus on evaluating the performance of the CNN-LSTM model in oil and gas field pressure prediction. The experimental results show that the CNN-LSTM model demonstrates significant advantages in the data prediction of multiple test Wells, and its prediction accuracy and stability are significantly superior to those of traditional machine learning models, showing stronger adaptability in multi-well pressure prediction in the study area. Specifically, the average R² value of the CNN-LSTM model in the 5-fold cross-validation reached 0.9982, which was much higher than that of other models, and its MSE and MAE are lowest compared with other models. It also has the smallest error in the pressure prediction of the other five Wells with unknown pressure, demonstrating excellent generalization ability.

By comparing with traditional machine learning models, the CNN-LSTM model can better capture the complex nonlinear relationships and temporal dependencies in the data, thereby providing more accurate and consistent pressure prediction results in the prediction of multiple Wells.

Code availability section

Name of the code/library: CNN_LSTM_Pp_pred.

Contact: e-mail: chenjunlin@cug.edu.cn.

Hardware requirements: Laptop or desktop PC.

Program language: the code is written in Python 3.13.

Software required: Python (3.13) and torch, scikit-learn and numpy packages.

Program size: 55 kb.

The source codes are available for downloading at the link: [https://github.com/GeoHydrocarbon/](https://github.com/GeoHydrocarbon/CNN_LSTM_Pp_pred)CNN_LSTM_Pp_pred.

Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

Acknowledgments

This work was supported by the National Science and Technology Major Project of the Chinese government. (No. 2024ZD1405101)

# Reference

An, Y., Du, H., Ma, S., Niu, Y., Liu, D., Wang, J., Du, Y., Childs, C., Walsh, J., Dong, R., 2023. Current state and future directions for deep learning based automatic seismic fault interpretation: A systematic review. Earth-Science Reviews 243, 104509. https://doi.org/10.1016/j.earscirev.2023.104509

Banerjee, A., Chatterjee, R., 2021. Pore pressure modeling and in situ stress determination in raniganj basin, India. Bull Eng Geol Environ 81, 49. https://doi.org/10.1007/s10064-021-02502-0

Bowers, G.L., 1995. Pore pressure estimation from velocity data: Accounting for overpressure mechanisms besides undercompaction. SPE Drill & Compl 10, 89–95. https://doi.org/10.2118/27488-PA

de Souza, J.A., Martínez, G.C., Castro Ponce de Leon, M.F., Azadpour, M., Atashbari, V., 2021. Chapter 14 - pore pressure and wellbore instability, in: Onajite, E. (Ed.), Applied Techniques to Integrated Oil and Gas Reservoir Characterization. Elsevier, pp. 355–394. https://doi.org/10.1016/B978-0-12-817236-0.00014-5

Eaton, B.A., 1975. The equation for geopressure prediction from well logs. Presented at the Fall Meeting of the Society of Petroleum Engineers of AIME, OnePetro. https://doi.org/10.2118/5544-MS

Frénay, B., Verleysen, M., 2011. Parameter-insensitive kernel in extreme learning for non-linear support vector regression. Neurocomputing, Advances in Extreme Learning Machine: Theory and Applications 74, 2526–2531. https://doi.org/10.1016/j.neucom.2010.11.037

Gilik, A., Ogrenci, A.S., Ozmen, A., 2022. Air quality prediction using CNN+LSTM-based hybrid deep learning architecture. Environ Sci Pollut Res 29, 11920–11938. https://doi.org/10.1007/s11356-021-16227-w

Greff, K., Srivastava, R.K., Koutník, J., Steunebrink, B.R., Schmidhuber, J., 2017. LSTM: A search space odyssey. IEEE Transactions on Neural Networks and Learning Systems 28, 2222–2232. https://doi.org/10.1109/TNNLS.2016.2582924

Hasan, Md.K., Jawad, Md.T., Dutta, A., Awal, Md.A., Islam, Md.A., Masud, M., Al-Amri, J.F., 2021. Associating Measles Vaccine Uptake Classification and its Underlying Factors Using an Ensemble of Machine Learning Models. IEEE Access 9, 119613–119628. https://doi.org/10.1109/ACCESS.2021.3108551

Henning Dypvik, 1983. Clay mineral transformations in tertiary and mesozoic sediments from north sea. Bulletin 67. https://doi.org/10.1306/03b5acdc-16d1-11d7-8645000102c1865d

Hua, Y., Guo, X., Tao, Z., He, S., Dong, T., Han, Y., Yang, R., 2021. Mechanisms for overpressure generation in the bonan sag of zhanhua depression, bohai bay basin, China. Marine and Petroleum Geology 128, 105032. https://doi.org/10.1016/j.marpetgeo.2021.105032

Hussain, W., Luo, M., Ali, M., Rizvi, S.N.R., Al-Khafaji, H.F., Ali, N., Ahmed, S.A.A., 2025. Advanced permeability prediction through two-dimensional geological feature image extraction with CNN regression from well logs data. Math Geosci 57, 657–702. https://doi.org/10.1007/s11004-024-10171-4

Jafarizadeh, F., Rajabi, M., Tabasi, S., Seyedkamali, R., Davoodi, S., Ghorbani, H., Alvar, M.A., Radwan, A.E., Csaba, M., 2022. Data driven models to predict pore pressure using drilling and petrophysical data. Energy Reports 8, 6551–6562. https://doi.org/10.1016/j.egyr.2022.04.073

Kim, K.G., 2016. Book review: Deep learning. Healthc Inform Res 22, 351. https://doi.org/10.4258/hir.2016.22.4.351

Lade, P.V., De Boer, R., 1997. The concept of effective stress for soil, concrete and rock. Géotechnique 47, 61–78. https://doi.org/10.1680/geot.1997.47.1.61

Lary, D.J., Alavi, A.H., Gandomi, A.H., Walker, A.L., 2016. Machine learning in geosciences and remote sensing. Geoscience Frontiers, Special Issue: Progress of Machine Learning in Geosciences 7, 3–10. https://doi.org/10.1016/j.gsf.2015.07.003

Liu, J., Liu, T., Liu, H., He, L., Zheng, L., 2021. Overpressure caused by hydrocarbon generation in the organic-rich shales of the ordos basin. Marine and Petroleum Geology 134, 105349. https://doi.org/10.1016/j.marpetgeo.2021.105349

Maritan, L., 2025. Machine learning models for bankruptcy prediction. Machine Learning Models for Bankruptcy Prediction.

Mark J. Osborne And Richard E. Swar, 1997. Mechanisms for generating overpressure in sedimentary basins: A reevaluation. Bulletin 81 (1997). https://doi.org/10.1306/522b49c9-1727-11d7-8645000102c1865d

McConnell, D.R., Zhang, Z., Boswell, R., 2012. Review of progress in evaluating gas hydrate drilling hazards. Marine and Petroleum Geology, Resource and hazard implications of gas hydrates in the Northern Gulf of Mexico: Results of the 2009 Joint Industry Project Leg II Drilling Expedition 34, 209–223. https://doi.org/10.1016/j.marpetgeo.2012.02.010

Nissa, N., Jamwal, S., Neshat, M., 2024. A Technical Comparative Heart Disease Prediction Framework Using Boosting Ensemble Techniques. Computation 12, 15. https://doi.org/10.3390/computation12010015

O’connor, S., Swarbrick, R., Lahann, R., 2011. Geologically-driven pore fluid pressure models and their implications for petroleum exploration. Introduction to thematic set. Geofluids 11, 343–348. https://doi.org/10.1111/j.1468-8123.2011.00354.x

Sahoo, B.B., Jha, R., Singh, A., Kumar, D., 2019. Long short-term memory (LSTM) recurrent neural network for low-flow hydrological time series forecasting. Acta Geophys. 67, 1471–1481. https://doi.org/10.1007/s11600-019-00330-1

Sarker, I.H., 2021. Machine learning: Algorithms, real-world applications and research directions. SN COMPUT. SCI. 2, 160. https://doi.org/10.1007/s42979-021-00592-x

Shi, W., Xie, Y., Wang, Z., Li, X., Tong, C., 2013. Characteristics of overpressure distribution and its implication for hydrocarbon exploration in the qiongdongnan basin. Journal of Asian Earth Sciences 66, 150–165. https://doi.org/10.1016/j.jseaes.2012.12.037

Stumpf, A., Kerle, N., 2011. Object-oriented mapping of landslides using random forests. Remote Sensing of Environment 115, 2564–2577. https://doi.org/10.1016/j.rse.2011.05.013

Sun, Y., Pang, S., Zhang, J., Zhang, Y., 2024. Porosity prediction through well logging data: A combined approach of convolutional neural network and transformer model (CNN-transformer). Physics of Fluids 36. https://doi.org/10.1063/5.0190078

Yu, H., Chen, G., Gu, H., 2020. A machine learning methodology for multivariate pore-pressure prediction. Computers & Geosciences 143, 104548. https://doi.org/10.1016/j.cageo.2020.104548

Zhang, G., Davoodi, S., Band, S.S., Ghorbani, H., Mosavi, A., Moslehpour, M., 2022. A robust approach to pore pressure prediction applying petrophysical log data aided by machine learning techniques. Energy Reports 8, 2233–2247. https://doi.org/10.1016/j.egyr.2022.01.012

Zhang, J., 2011. Pore pressure prediction from well logs: Methods, modifications, and new approaches. Earth-Science Reviews 108, 50–63. https://doi.org/10.1016/j.earscirev.2011.06.001

Zhang, W., Wu, C., Liu, S., Liu, X., Wu, X., Lu, X., 2024. Impact of disequilibrium compaction and unloading on overpressure in the southern junggar foreland basin, NW China. Marine and Petroleum Geology 164, 106819. https://doi.org/10.1016/j.marpetgeo.2024.106819

Zhang, W., Zhou, H., Bao, X., Cui, H., 2023. Outlet water temperature prediction of energy pile based on spatial-temporal feature extraction through CNN–LSTM hybrid model. Energy 264, 126190. https://doi.org/10.1016/j.energy.2022.126190

Zhao, J., Li, J., Xu, Z., 2018. Advances in the origin of overpressures in sedimentary basins. Petroleum Research 3, 1–24. https://doi.org/10.1016/j.ptlrs.2018.03.007

Zheng, R., Huang, M.C., 2017. Redundant memory array architecture for efficient selective protection, in: 2017 ACM/IEEE 44th Annual International Symposium on Computer Architecture (ISCA). Presented at the 2017 ACM/IEEE 44th Annual International Symposium on Computer Architecture (ISCA), pp. 214–227. https://doi.org/10.1145/3079856.3080213

|     |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | DT  | RF  | XGBoost | CatBoost | MLP | SVR | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | Full Test |
| MSE | 10.48 | 6.26 | 6.36 | 7.09 | 6.97 | 8.05 | 0.22 | 0.46 | 0.22 | 0.35 | 0.42 | 0.61 |
| MAE | 1.90 | 1.37 | 1.50 | 1.72 | 1.71 | 1.84 | 0.37 | 0.50 | 0.35 | 0.44 | 0.48 | 0.59 |

Table 2. MSE and MAE for each ML and CNN-LSTM (5 folds and full test) models.

|     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Well<br><br>name | Depth(m) | Measured<br><br>pressure (MPa) | Errors of ML models (%) |     |     |     |     |     | Errors of CNN-LSTM model of 5-fold and full data (%) |     |     |     |     |     |
| **DT** | **RF** | **MLP** | **XGBoost** | **CatBoost** | **SVR** | **Fold1** | **Fold2** | **Fold3** | **Fold4** | **Fold5** | **Full Test** |
| Well 2 | 3042.4 | 34.01 | 9.499 | 12.912 | 11.733 | 10.221 | 9.524 | 5.793 | 1.184 | 4.195 | 15.218 | 1.009 | 2.372 | 1.069 |
| Well 2 | 3275.1 | 48.16 | 5.671 | 4.829 | 14.551 | 2.401 | 2.584 | 3.769 | 2.602 | 3.099 | 0.493 | 1.014 | 3.860 | 3.302 |
| Well 3 | 3025.05 | 28.43 | 16.365 | 12.712 | 3.875 | 13.335 | 10.535 | 2.092 | 2.899 | 2.077 | 0.497 | 8.405 | 5.502 | 7.233 |
| Well 3 | 3394.9 | 56.37 | 54.586 | 47.090 | 1.183 | 40.896 | 28.579 | 11.638 | 2.178 | 0.942 | 1.866 | 1.794 | 0.037 | 2.522 |
| Well 3 | 3494.4 | 60.74 | 30.509 | 19.764 | 16.802 | 15.099 | 14.213 | 18.554 | 13.514 | 10.749 | 11.667 | 10.943 | 12.697 | 12.054 |
| Well 4 | 3170.3 | 48.68 | 14.429 | 18.395 | 22.033 | 17.920 | 19.737 | 24.127 | 10.478 | 9.540 | 12.397 | 9.700 | 11.678 | 18.230 |
| Well 5 | 3285.3 | 42.37 | 33.443 | 32.119 | 1.058 | 30.553 | 24.838 | 2.872 | 12.976 | 12.661 | 17.410 | 19.607 | 15.851 | 13.813 |
| Well 6 | 3422.4 | 44.47 | 27.652 | 23.656 | 41.617 | 16.888 | 22.108 | 0.708 | 7.386 | 2.815 | 0.758 | 8.793 | 1.569 | 5.045 |
| Average error (%) |     |     | 24.019 | 21.435 | 14.107 | 18.414 | 16.515 | 8.694 | 6.652 | 5.760 | 7.538 | 7.658 | 6.696 | 7.909 |

Table 3. Errors of predicted pressure and measured pressure for each ML and CNN-LSTM (5 folds and full test) models.
