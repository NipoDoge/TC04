import numpy as np
import pandera.pandas as pa

LATE = ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTimes90DaysLate",
        "NumberOfTime60-89DaysPastDueNotWorse"]
COUNTS = ["NumberOfOpenCreditLinesAndLoans", "NumberRealEstateLoansOrLines"]

columns = {
    "SeriousDlqin2yrs": pa.Column(int, pa.Check.isin([0, 1])),            # target binário
    "age": pa.Column(int, pa.Check.in_range(18, 100)),                    # maioridade civil
    "MonthlyIncome": pa.Column(float, pa.Check.ge(0), nullable=False),    # renda obrigatória
    "DebtRatio": pa.Column(float, pa.Check.ge(0)),                        # razão não negativa
    "RevolvingUtilizationOfUnsecuredLines": pa.Column(float, pa.Check.ge(0)),
    "NumberOfDependents": pa.Column(float, pa.Check.ge(0)),
    **{c: pa.Column(int, pa.Check.ge(0)) for c in LATE + COUNTS},
}

# A lista também define a ordem de entrada usada pelo modelo.
FEATURE_COLUMNS = [name for name in columns if name != "SeriousDlqin2yrs"]

for column in columns.values():
    column.checks.append(pa.Check(np.isfinite, name="finite"))

schema = pa.DataFrameSchema(
    columns,
    checks=pa.Check(lambda df: not df.empty, name="non_empty"),
    strict=True,                 # coluna inesperada também reprova
    unique=list(columns),        # linhas duplicadas reprovam
)
