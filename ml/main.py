from utils import load_dataset
from tests import test1

enable_prints = False

print("Obtenemos el dataset con las variables que vamos a usar, ya transformadas")
df = load_dataset(enable_prints)

print("Experimentos")
print("Experimento con todos los campos")
test1(df)

print("end")
