Project overview：
This is an agricultural data analysis project implemented in Python. The core is to quantify the impact of agricultural inputs (irrigation, fertilizer and agricultural machinery) on grain yield in Hunan Province from 2014 to 2023 through econometric models, and finally output implementable input optimization suggestions.
The code covers the whole process of "data cleaning, modeling and analysis, result visualization and report export", which is suitable for agricultural data analysis reference. It can also be directly reused and modified for agricultural analysis in other provinces/regions.

Why is the project?
1,Hunan province is the main grain producing area in southern China (rice accounts for 12% of the country), but there is no quantitative data to support the efficiency of agricultural input
2,In the national policy of "reducing fertilizer and increasing efficiency" and "promoting agricultural machinery", data is needed to verify the actual effect
3,As a student majoring in rural regional development, I hope to use technology to solve the practical problem of "input priority" in agricultural production

The problem that was hoped to be solved:
1️⃣Which of the three types of input, irrigation area, fertilizer consumption and agricultural machinery power, has the greatest impact on grain yield?
2️⃣Do we still need to increase fertilizer production?
3️⃣How should agricultural investment be allocated to balance increased production and green development?

Data preparation: Read CSV → rename variables (Y= yield, X1= irrigation, X2= fertilizer, X3= agricultural machinery)
Data cleaning: check for missing values → 3σ outlier test → standardization (e.g. fertilizer conversion treatment)
Descriptive analysis: calculate mean/standard deviation → draw trend chart and heat map
Measurement modeling: construct OLS regression model → calculate coefficients and significance
Model diagnosis: VIF test (collinearity) → White test (heteroskedasticity) → DW test (autocorrelation)
Result output: Export Excel report → Save visual chart

Source: China National Bureau of Statistics website
