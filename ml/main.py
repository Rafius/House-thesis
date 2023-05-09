from utils import *
from tests import *
import time

start_time = time.time()

enable_prints = False

# print("Obtenemos el dataset con las variables que vamos a usar, ya transformadas")
df = load_dataset(enable_prints)

# print("Experimentos")
# test1(df)
# test2(df)
# test3(df)
# test4(df)
# test5(df)
# test6(df)
# test7(df)
# test8(df)
# test9(df)
# test10(df)

base_test(df)

estimate_houses_to_buy_rent_prices()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"El tiempo total de ejecución fue de {elapsed_time:.2f} segundos")