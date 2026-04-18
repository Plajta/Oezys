import numpy as np
import matplotlib.pyplot as plt

# 1. Načti to jako syrová data
# Zkus 'int16', 'int32' nebo 'float32' - jeden z nich to bude
raw_data = np.fromfile('./models/TRAIN_SET/ZdraviLudia/Kontr_01.03.2023_LO.016', dtype=np.int16)

print(f"Celkový počet bodů: {len(raw_data)}")

# 2. Hledáme rozlišení (musí to být čtverec nebo známý poměr)
# Tip: zkus odmocninu z celkového počtu bodů
side = int(np.sqrt(len(raw_data)))
if side * side == len(raw_data):
    print(f"Vypadá to na čtverec: {side}x{side}")
    matrix = raw_data.reshape((side, side))
    plt.imshow(matrix, cmap='hot')
    plt.show()
else:
    # Někdy je tam na začátku 'offset' (hlavička), zkus ji přeskočit
    # Zkus smazat prvních X bajtů, dokud to nevyjde
    print("Není to čistý čtverec, asi to má hlavičku.")