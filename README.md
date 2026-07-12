# Walmart Recruiting: Store Sales Forecasting

## კონკურსის მიმოხილვა

[Walmart Recruiting: Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting) კონკურსის მიზანია Walmart-ის 45 მაღაზიის სხვადასხვა დეპარტამენტის კვირეული გაყიდვების პროგნოზირება ისტორიული მონაცემების საფუძველზე. ამოცანა ფასდება WMAE (Weighted Mean Absolute Error) მეტრიკით predicted და რეალურ გაყიდვებს შორის, სადაც სადღესასწაულო კვირებს 5-ჯერ მეტი წონა აქვთ. შესაბამისად, დღესასწაულების ზუსტი პროგნოზირება ამ ამოცანაში განსაკუთრებით მნიშვნელოვანია.

**მასშტაბი:** ~3,300 დროითი მწკრივი (`Store × Dept`), 421,570 სატრენინგო და 115,064 სატესტო row.

> ვინაიდან WMAE დღესასწაულებს 5× წონას აძლევს, **ყველა მოდელში loss function გავათანაბრეთ ამ მეტრიკასთან** - tree-based მოდელებში `objective="reg:absoluteerror"` / `"regression_l1"` (L1 = MAE) + `holiday_weight=5.0` sample weight; DL მოდელებში `L1Loss` + 5× holiday sample weight; TFT-ში quantile loss q=0.50 (median = L1). ამის შედეგად სატრენინგო loss პირდაპირ WMAE-ს ემთხვევა.

### რეპოზიტორიის სტრუქტურა

```
ML_FINAL_PROJECT/
├── data/                              # Kaggle-ის raw მონაცემები
│   ├── train.csv.zip
│   ├── test.csv.zip
│   ├── features.csv.zip
│   └── stores.csv
├── src/                              # საერთო, shared მოდულები
│   ├── preprocessing.py               # Cleaner - raw data cleaning
│   ├── feateng.py                     # FeatureBuilder, FEATURE_GROUPS 
│   ├── pipeline.py                    # WalmartPipeline (tree-based)
│   ├── ts_pipeline.py                 # GlobalDartsPipeline  
│   ├── ts_data.py                     # series building, covariates,
│   ├── validation.py                  # FOLDS, evaluate(), wmae()
│   └── experiment_utils.py            # setup_mlflow, run_stage, 
├── notebooks/
├── model_experiment_XGBoost.ipynb     # Tree-Based
├── model_experiment_LightGBM.ipynb    # Tree-Based
├── model_experiment_DLinear.ipynb     # Deep Learning
├── model_experiment_NBEATS.ipynb      # Deep Learning
├── model_experiment_TFT.ipynb         # Deep Learning
├── model_experiment_Prophet.ipynb     # Classical Statistical
├── model_experiment_ARIMA.ipynb       # Classical Statistical
├── model_experiment_SARIMA.ipynb      # Classical Statistical
├── model_inference.ipynb              # საბოლოო submission Model 
├──
├── submissions/                       # Kaggle submission ფაილები
└── README.md
```

### ფაილების განმარტება

| ფაილი | მიზანი |
|---|---|
| `model_experiment_{architecture}.ipynb` | თითო არქიტექტურის სრული workflow: Preprocessing → Feature Engineering → Feature Selection → CV → Final |
| `model_inference.ipynb` | MLflow Model Registry-დან საუკეთესო მოდელის ჩამოტვირთვა და საბოლოო Kaggle submission |
| `src/` | ყველა notebook-ის მიერ გაზიარებული preprocessing, feature engineering, validation და pipeline მოდულები |
| `README.md` | პროექტის სრული დოკუმენტაცია |

### MLflow Tracking

ყველა ექსპერიმენტი დალოგილია **DagsHub MLflow**-ზე:
**https://dagshub.com/ZukaCS/ML_FINAL_PROJECT.mlflow**

თითოეულ არქიტექტურას აქვს **ცალკე Experiment** (მაგ. `XGBoost_Training`), რომლის შიგნითაც stage-ების მიხედვით არსებობს ცალკე run-ები (`Preprocessing`, `Feature_Engineering`, `Feature_Selection`, `CV`, `Final`) - CV stage-ში კი თითო nested child run თითო კონფიგურაციაზე.

---

## 2. მონაცემთა წინასწარი დამუშავება და Feature Engineering

### 2.1. საერთო EDA და Cleaning


| observation | მნიშვნელობა | გავლენა მოდელირებაზე |
|---|---|---|
| **სეზონურობა (m = 52)** | ყოველწლიური ციკლი. Thanksgiving/Christmas-ის მკვეთრი პიკები | ცენტრალური სიგნალი. ამის დაჭერა განაპირობებს მოდელის ხარისხს |
| **მწკრივების სიგრძე** | 143 კვირა = **მხოლოდ 2.75 წლიური ციკლი** | სტატისტიკურად "თხელი" seasonal estimation |
| **negative sales (~0.3%)** | დაბრუნებები (returns), median ≈ −$1 | clip to 0 |
| **skewness = 3.26** | გაყიდვების right-skewed განაწილება | `log1p` შემოწმდა A/B-ით ხეებში ხოლო სხვა მოდელებში target უმეტესად გავალოგარითმეთ|
| **MarkDown 1-5** | sparse, მხოლოდ 2011 ნოემბრის შემდეგ | zero-impute; მცირე სიგნალი |
| **pre-Christmas week** | `IsHoliday` **ვერ ხედავს** შობის წინა კვირის პიკს | Prophet-ის holidays table-ში ხელით დაემატა (EDA finding) |

**shared reshaping (`Cleaner` + `build_target_series`):**

- ყველა მწკრივი დაყვანილია სტანდარტულ კვირეულ `W-FRI` ბადეზე (კვირა პარასკევს სრულდება);
- შიდა gap-ები (`n_gap_weeks_filled = 27,667`) შევსებულია 0-ით;
- 372 მწკრივს ბოლო ნაწილი (tail) 0-ით padded არის საერთო ბოლო თარიღამდე, რომ `predict(n)` ყველა წყვილისთვის ერთ კალენდარულ თარიღზე დაჯდეს;
- `MarkDown → 0`, `CPI/Unemployment → ffill` მაღაზიის დონეზე;
- 376 late-start მწკრივი შენარჩუნებულია backfill-ის გარეშე;
- მოდელისთვის (short/gappy) მწკრივებისთვის **seasonal-naive fallback** (department median).

### 2.2. Feature Engineering (Tree-Based მოდელები)

Feature engineering-ის ძირითადი სამუშაო **tree-based** მოდელებზე მოდის - მათ სჭირდებათ tabular features. ეს გაზიარებულია `src/feateng.py`-ის `FeatureBuilder`-ში (`FEATURE_GROUPS`):

| ჯგუფი | features | დანიშნულება |
|---|---|---|
| **`lags`** | `lag_52` (ზუსტად 364 დღით ადრინდელი გაყიდვა) და მოკლე lag-ები | **ცენტრალური feature** - გასულ წელს იმავე კვირის გაყიდვა პირდაპირ ატარებს დღესასწაულის პიკს |
| **`lag_windows`** | rolling window statistics (mean/std) | rolling mean (ბოლო N კვირის საშუალო) და rolling std (ბოლო N კვირის სტანდარტული გადახრა) |
| **`calendar`** | კვირა/თვე/წელი, week-of-year | კალენდარული პოზიცია |
| **`holiday`** | დღესასწაულის ინდიკატორები/მანძილები | holiday effect |
| **`statics`** | Store Type, Size (stores.csv-დან) | მაღაზიის კონტექსტი |

> **Data Leakage-ის თავიდან აცილება:** `FeatureBuilder`-ის fit ხდება მხოლოდ fold-ის **სატრენინგო** slice-ზე. ორი sanity assert ამას ადასტურებს: (ა) `lag_52` ზუსტად 364 დღით ადრინდელ მნიშვნელობას აბრუნებს (მხოლოდ warmup), (ბ) fold-1-ის builder-ს არაფერი "ახსოვს" fold boundary-ის შემდეგ. `lag_52`-ის coverage სატრენინგოზე 62%-ია (პირველი წელი warmup-ია).

### 2.3. Feature Engineering (DL და Classical მოდელები)

აქ არქიტექტურა თავად განსაზღვრავს features-ს:

- **DLinear / N-BEATS** - **univariate**, paper-native: მხოლოდ target series. სეზონურობა input window-ში (52-65 კვირა) უნდა ჩაეტიოს.
- **TFT** - **covariate-rich**: სრული future covariate set (calendar, holiday distances, markdowns, CPI, temperature, fuel, unemployment) + static covariates (Store, Dept, Type, Size). Store/Dept - **categorical embeddings**.
- **Prophet** - trend + yearly Fourier seasonality + **holidays table** (4 დასახელებული დღესასწაული + pre-Christmas peak).
- **ARIMA / SARIMA** - **წმინდა univariate**, endogenous: სეზონურობა SARIMA-ში seasonal differencing-ით მოდელირდება, ARIMA-ში კი საერთოდ არ არსებობს. Feature engineering სტრუქტურულად არარელევანტურია (baseline).

### 2.4. Cross-Validation - Time-Ordered, No Leakage

ყველა global მოდელი ფასდება **3-fold rolling-origin (walk-forward) CV**-ით (`src/validation.py`):

- **დროის მიხედვით დაყოფა** (არა random KFold) - validation ყოველთვის train-ის შემდეგ მოდის;
- **Fold 1** მოიცავს Thanksgiving + Christmas 2011-ს - ეს არის fold, რომელიც **leaderboard-ს ყველაზე კარგად წინასწარმეტყველებს**;
- Fold 2, 3 თანდათან უფრო "მარტივია" (ნაკლები holiday stress).

> **მნიშვნელოვანი მეთოდოლოგიური დასკვნა:** `wmae_mean` ოპტიმისტურია, რადგან fold-3 იოლია. **Fold-1 WMAE** არის რეალური leaderboard proxy - ამას ქვემოთ, შედეგების ანალიზში დავადასტურებთ ციფრებით.

---

## 3. ექსპერიმენტები და მოდელების არქიტექტურა

MLflow-ის სტრუქტურა ყველა არქიტექტურაზე ერთგვაროვანია: **Experiment = `{Arch}_Training`**, შიგნით stage run-ები. ქვემოთ თითოეული ოჯახი განხილულია ცალკე.

---

### 3.1. Tree-Based Models - XGBoost & LightGBM

#### Architecture Overview

Gradient Boosting აშენებს **additive trees ensembele-ს**, სადაც თითო ხე წინა ხეების residual-ს ასწორებს. Time-series-ისთვის ხეები **პირდაპირ არ ხედავენ დროს** - ამიტომ დროითი დამოკიდებულება მათ `lag_52`-ისა და calendar features-ის სახით მიეწოდებათ. ორივე მოდელი **global**-ია: ერთი მოდელი სწავლობს ~3,300-ვე მწკრივს ერთად, რაც მწკრივებს შორის ინფორმაციის გაზიარებას (cross-series learning) იძლევა - ეს კრიტიკულია იმ მწკრივებისთვის, სადაც ისტორია მცირეა.

- **XGBoost** - `tree_method="hist"`, `device="cuda"`, `enable_categorical=True`, `objective="reg:absoluteerror"`.
- **LightGBM** - leaf-wise growth (`num_leaves`), `objective="regression_l1"`, ჩვეულებრივ უფრო სწრაფი.


#### Hyperparameters & Tuning

Optuna **TPE sampler**, 30 trial, objective = `wmae_mean`:

| XGBoost | LightGBM |
|---|---|
| `n_estimators` (300-1200) | `n_estimators` (300-1200) |
| `learning_rate` (0.03-0.2, log) | `learning_rate` (0.03-0.2, log) |
| `max_depth` (4-11) | `num_leaves` (31-255, log) |
| `min_child_weight` (1-100, log) | `min_child_samples` (10-100, log) |
| `colsample_bytree`, `subsample` (0.6-1.0) | `colsample_bytree`, `subsample` (0.6-1.0) |
| `reg_alpha`, `reg_lambda` (1e-3-10, log) | `reg_alpha`, `reg_lambda` (1e-3-10, log) |

**Preprocessing A/B-ის შედეგი (ორივე მოდელი):** გამარჯვებული `clip_neg__raw` - negatives clip 0-მდე, raw target. `log1p`-მა შედეგი **გააუარესა** (XGBoost `wmae_mean` 1968 → 2059), რაც ადასტურებს, რომ trees skew-ის მიმართ მდგრადია და L1 loss log-transform-ს არ საჭიროებს.

> **Underfit/Overfit ანალიზი (train vs val gap):** XGBoost - `train_wmae=926`, `val_wmae=1792`, **gap=866** (მაღალი capacity, მეტ overfit-ს იძლევა, მაგრამ საუკეთესო generalization). LightGBM - `train=1570`, `val=1898`, **gap=328** (უფრო regularized). საინტერესოა LightGBM-ის fold-3, სადაც gap **უარყოფითია (−155)**: სატრენინგო holiday კვირები 5× წონით უფრო დიდ error-ს იძლევა, ვიდრე fold-3-ის (holiday-ს გარეშე) validation.

---

### 3.2.  Deep Learning - DLinear, N-BEATS, TFT

> სამივე DL მოდელი აშენებულია **Darts**-ზე, `GlobalDartsPipeline`-ით (ერთი global მოდელი ყველა მწკრივზე), hyperparameter tuning-ით **Optuna TPE**.

#### Architecture Overview

| მოდელი | ტიპი | არქიტექტურა | Covariates |
|---|---|---|---|
| **DLinear** (Zeng et al. 2022) | linear, univariate | series-ის trend/remainder **დეკომპოზიცია** + თითო კომპონენტზე ერთი linear layer | არა (paper-native) |
| **N-BEATS** (Oreshkin et al. 2020) | nonlinear, univariate | FC ბლოკების stack + **basis expansion** + double residual; generic ან interpretable (trend+seasonality) | არა |
| **TFT** (Lim et al. 2021) | attention, covariate-rich | LSTM encoder/decoder + **variable selection networks** + gated residual + interpretable multi-head attention | დიახ (სრული set + statics) |

ეს სამი ქმნის **სუფთა კონტრასტს**: DLinear = univariate linear, N-BEATS = univariate nonlinear, TFT = covariate-rich attention.

#### Hyperparameters & Tuning

| მოდელი | trials | loss / likelihood | ძირითადი search space |
|---|---|---|---|
| **DLinear** | 30 | `L1Loss` + 5× holiday weight = WMAE | input/output chunk (26-65 / 13-39), `kernel_size` (13/25/51), `const_init`, `batch_size`, `lr` |
| **N-BEATS** | 25 | `L1Loss` + 5× holiday weight | chunk pairs, `generic` vs `interpretable`, `num_stacks/blocks/layers`, `layer_widths` (128-512), `dropout`, `lr`; `max_samples_per_ts=10` |
| **TFT** | 12 | **Quantile Regression**, q=0.50 median (= L1 = WMAE) | chunk (52/65), `hidden_size` (16-64), `lstm_layers`, `num_attention_heads`, `dropout`, `covariate_preset` (`calendar_holiday` vs `full`), `lr` |

> **TFT-ის დახვეწილი დეტალი:** predict-ისას pipeline **quantile PARAMETERS**-ს (deterministic) ითხოვს და q0.50 median-ს იტოვებს - ეს არის WMAE-ოპტიმალური point forecast (`num_samples=1` სტოქასტურ sample-ს დახატავდა). Store/Dept შედის **categorical embeddings**-ად (nominal id-ები scale-ულ რიცხვებად ვერ შევა).

---

### 3.3.  Classical Statistical - ARIMA, SARIMA, Prophet

> assignment-ის მიხედვით, კლასიკური მოდელები **თეორიულად უფრო მნიშვნელოვანია**, და ამ მოდელების trainingზე შედარებით ნაკლები დრო დავხარჯეთ. ARIMA/SARIMA აშენებულია პირდაპირ **`statsmodels`**-ზე (self-contained, `darts` wrapper-ის გარეშე) - ეს იძლევა უზარმაზარ სიჩქარეს: order-ის შერჩევა ხდება **ერთხელ თითო მწკრივზე** AICc-ით.

#### Architecture Overview

| მოდელი | არქიტექტურა | სეზონურობა |
|---|---|---|
| **ARIMA** (p,d,q) | წმინდა non-seasonal AR + differencing + MA | **არა** - honest baseline |
| **SARIMA** (p,d,q)(P,D,Q)[52] | + seasonal differencing და seasonal AR/MA | **დიახ**, endogenous, m=52 |
| **Prophet** (Taylor & Letham 2017) | piecewise-linear trend + yearly Fourier seasonality + holidays | დიახ (Fourier + holidays table) |

#### MLflow Experiment Structure & მიდგომა

ორივე (ARIMA, SARIMA) იყენებს **leakage-free** მიდგომას: order-ს ირჩევს მხოლოდ **ყველაზე ადრინდელი CV fold-ის** სატრენინგო window-ზე, pooled WMAE-ით (არა per-series საშუალო). შერჩევა ხდება **30 representative** (stratified, complete) მწკრივზე, შემდეგ საუკეთესო order მოწმდება **ყველა ~3,300 მწკრივზე** (`full_holdout_wmae`).

- **ARIMA_CV** - 4 trial: no-exog / IsHoliday exog / full economic ARIMAX / wide order grid. გამარჯვებული: `D_no_exog_wide_grid`, order **(2,1,3)**.
- **SARIMA_CV** - 3 trial: airline `(0,1,1)(0,1,1)` / **full base grid 64 combos** `(p,d,q)×(P,D,Q)` / wide grid 144 combos. გამარჯვებული: `C_wide_grid`, order **(0,1,1)(0,1,0)[52]**.
- **Prophet_CV** - 24 Optuna trial stratified 300-sample-ზე, შემდეგ full 3-fold.

#### Prophet-ის feature-ორიენტირებული ხერხი

Prophet-ს მიეწოდება **ხელით აგებული holidays table**: ოთხი დასახელებული holiday კვირა + **pre-Christmas peak week**, რომელსაც `IsHoliday` **ვერ ხედავს** (EDA finding). ეს პირდაპირ WMAE-კრიტიკულ კვირებზე მუშაობს. Tuned params: `changepoint_prior_scale`, `seasonality_prior_scale`, `holidays_prior_scale`, `seasonality_mode` (additive/multiplicative), `yearly_seasonality` (6/10/20), `n_changepoints`.



## 4. მოდელების შეფასება და შედეგების შედარება

### 4.1. საბოლოო შედეგები (Kaggle Public Score-ით დალაგებული)

| # | მოდელი | კატეგორია | Internal WMAE | Fold-1 WMAE | **Kaggle Public** | Kaggle Private |
|---|---|---|---|---|---|---|
| 🥇 | **XGBoost** | Tree | 1792 (CV mean) | 2346 | **2628.71** | 2745.08 |
| 🥈 | **LightGBM** | Tree | 1898 (CV mean) | 2563 | 2743.33 | 2863.60 |
| 🥉 | **Prophet** | Classical | 1741 (CV mean) | 2145 | 2767.63 | 2869.75 |
| 4 | **DLinear** | DL | 2105 (CV mean) | 3063 | 2877.71 | 3028.63 |
| 5 | **SARIMA** | Classical | 1553 (full holdout) | - | 2897.09 | 2994.73 |
| 6 | **N-BEATS** | DL | 2221 (CV mean) | 3204 | 3368.29 | 3499.63 |
| 7 | **TFT** | DL | 3263 (CV mean) | 4631 | 4509.00 | - |
| 8 | **ARIMA** | Classical | 2056 (full holdout) | - | 4570.10 | 4822.89 |

> **საუკეთესო მოდელი: XGBoost - Kaggle Public WMAE = 2628.71.** ეს მოდელია registered Model Registry-ში და გამოიყენება `model_inference.ipynb`-ში.

*(შენიშვნა: ARIMA/SARIMA-ს "Internal WMAE" არის `full_holdout_wmae` სხვა CV პროტოკოლით - ამიტომ არ არის პირდაპირ შესადარებელი global მოდელების `wmae_mean`-თან. ერთადერთი მთლიანად სამართლიანი შედარება Kaggle score-ია.)*

### 4.2. სიღრმისეული ანალიზი - რატომ გაიმარჯვა XGBoost-მა?

**1. Global learning + `lag_52` = generalization.** XGBoost/LightGBM ერთ მოდელში სწავლობენ ~3,300-ვე მწკრივს და იყენებენ `lag_52`-ს (გასული წლის იმავე კვირის გაყიდვა). ეს feature **პირდაპირ ატარებს გასული წლის holiday პიკს** - ზუსტად ის სიგნალი, რომელიც სატესტო 39-კვირიან horizon-ს (2012 Nov - 2013) სჭირდება. ლოკალური მოდელები (ARIMA/SARIMA/Prophet) კი თითო მწკრივს **იზოლირებულად** სწავლობენ და cross-series ინფორმაციას ვერ იყენებენ.


**3. რატომ ჩამორჩა კლასიკური სტატისტიკა?**
- **ARIMA (worst, 4570)** - non-seasonal ARIMA-ს **არ აქვს მექანიზმი 52-კვირიანი ციკლისთვის**; horizon-ის უმეტესობა მისი მეხსიერების მიღმაა. ეს honest baseline-ია, "იატაკი", რომელსაც სხვები აჯობებენ.
- **SARIMA (2897)** - სეზონურობას **ხედავს** და ARIMA-ს მკვეთრად სჯობს (4570 → 2897), მაგრამ 143 კვირა = მხოლოდ 2.75 ციკლი → seasonal estimation თხელია.
- **Prophet (2767, საუკეთესო კლასიკური)** - holidays table + Fourier seasonality holiday კვირებზე კარგად მუშაობს, მაგრამ ლოკალურობა cross-series სწავლას ხელს უშლის.

**4. რატომ ჩამორჩა Deep Learning tree-ებს? (და "Are Transformers Effective?")**
DL მოდელების რანჟირება: **DLinear (2877) > N-BEATS (3368) > TFT (4509)** - ანუ **უფრო მარტივი მოდელი უკეთესია**. ეს პირდაპირ ადასტურებს DLinear-ის paper-ის თეზისს (*"Are Transformers Effective for Time Series Forecasting?"* - პასუხი: არა, ამ მასშტაბზე). მიზეზები:
- **მონაცემი მცირეა transformer-ისთვის** - 2.75 წელი TFT-ის capacity-სთვის საკმარისი არ არის;
- **TFT undertrained/overfit** - მძიმე მოდელი, ცოტა epoch (15), ცოტა trial (12), `max_samples_per_ts=10`;
- **covariates მცირე სიგნალს ამატებს** - markdown/macro features-ის მნიშვნელობა tree importance-შიც დაბალი აღმოჩნდა.

> **მთავარი takeaway:** ამ ამოცანაზე **feature engineering + global tree model** აჯობა როგორც კლასიკურ სტატისტიკას, ისე მძიმე deep learning-ს. `lag_52` (გასული წლის სეზონურობა) ცალკე უფრო ღირებული სიგნალი აღმოჩნდა, ვიდრე TFT-ის attention და covariate-ების მთელი აპარატი.





### დასკვნა

ARIMA-დან (Kaggle 4570) დახვეწილ gradient boosting-მდე (XGBoost, **2628**) WMAE **~42%-ით** გაუმჯობესდა. 

1. **Global tree model + `lag_52`** აჯობა როგორც კლასიკურ სტატისტიკას, ისე მძიმე deep learning-ს - ამ ამოცანაზე გასული წლის სეზონურობის ცალკე feature-ად მიწოდება უფრო ღირებული აღმოჩნდა, ვიდრე transformer-ის სირთულე.
2. **Fold-1 (holiday fold)** ერთადერთი სანდო leaderboard proxy-ია - `wmae_mean` ოპტიმისტურია.
3. DLinear > N-BEATS > TFT ადასტურებს, რომ მცირე მონაცემზე transformer-ის capacity არ ამართლებს.
4. **Loss = Metric** - L1 + 5× holiday weight-ის გათანაბრება WMAE-სთან ყველა ოჯახში კონსისტენტურად დაეხმარა holiday fold-ს.

---

*MLflow: [ZukaCS/ML_FINAL_PROJECT](https://dagshub.com/ZukaCS/ML_FINAL_PROJECT.mlflow) · Kaggle: Walmart Recruiting - Store Sales Forecasting*

