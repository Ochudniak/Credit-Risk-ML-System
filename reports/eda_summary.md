# EDA Summary

## Dataset loading

The dataset was loaded from an XLS file using `header=1`, because the first row contains technical column labels and the second row contains the real feature names.

## Dataset size

The dataset contains 30,000 rows and 25 columns.

## Target distribution

The target variable is `default payment next month`.

- 0 - no default: 23,364 rows (77.88%)
- 1 - default: 6,636 rows (22.12%)

The target is imbalanced. Accuracy alone may be misleading, because a naive model predicting only the majority class would already achieve around 77.88% accuracy.

## Missing values

No missing values were found in the dataset.

## Duplicate rows

No duplicate rows were found.

## Category values

`SEX` contains only values 1 and 2.

`EDUCATION` contains suspicious values 0, 5 and 6. 

`MARRIAGE` contains suspicious value 0. 

Values in Education and Marriage should be investigated and handled during the cleaning stage.

## Numeric ranges

The age range looks reasonable.

The credit limit range looks reasonable.

`BILL_AMT1` to `BILL_AMT6` contain negative values. These may represent overpayment or credit balance, so they should not be removed automatically.

`PAY_AMT1` to `PAY_AMT6` do not contain negative values.

## Initial conclusions

The dataset is usable for binary classification, but it requires cleaning before model training.

The next stage should focus on:
- removing or ignoring the `ID` column
- renaming the target column to a simpler name
- handling suspicious categories in `EDUCATION` and `MARRIAGE`
- keeping negative `BILL_AMT` values unless further analysis proves they are invalid
- preparing a clean dataset for modeling