from utils import load_dataset
from tests import *

enable_prints = False

# print("Obtenemos el dataset con las variables que vamos a usar, ya transformadas")
df = load_dataset(enable_prints)

# print("Experimentos")
# print("Experimento con todos los campos")
# test1(df)
# test2(df)
# test3(df)
test4(df)
# test5(df)
# test6(df)
# test6(df)
# test8(df)
print("end")
