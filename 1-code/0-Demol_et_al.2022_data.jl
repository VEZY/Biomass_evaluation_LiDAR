
# This script can be used to compute the errors from the paper of Demol et al. (2022)
using CSV, DataFrames, Statistics

df = CSV.read("0-data/0-raw/Demol_et_al.2022_data.csv", DataFrame)
transform!(df, [:volume_l, :QSM_l] => ByRow((x, y) -> (y - x) / x * 100) => :error)

# Errors for structures above 10 cm diameter:
df_sup10 = combine(
    filter(x -> x.diameter_class_cm == ">10 cm", df),
    :volume_l => mean => :volume_l,
    :QSM_l => mean => :QSM_l
)
transform(df_sup10, [:volume_l, :QSM_l] => ByRow((x, y) -> (y - x) / x * 100) => :error)

# Errors for structures below or equal to 10 cm diameter:
df_inf10 = combine(
    filter(x -> x.diameter_class_cm in ["0-2.5 cm", "2.5-5 cm", "5-7 cm", "7-10 cm"], df),
    :volume_l => mean => :volume_l,
    :QSM_l => mean => :QSM_l
)
transform(df_inf10, [:volume_l, :QSM_l] => ByRow((x, y) -> (y - x) / x * 100) => :error)
