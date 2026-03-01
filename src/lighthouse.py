import time
import hashlib

ALPHA_INV = 137.036108
WINDOW = 500 # Окно сканирования фазы в миллисекундах

def get_act_code(t_ms):
    state = float(t_ms) * ALPHA_INV
    return hashlib.sha256(str(state).encode()).hexdigest()[-4:].upper()

print("--- ACT PHASE SCANNER STARTING ---")

try:
    while True:
        t_now = int(time.time() * 1000)
        
        # Наш текущий локальный код
        local_code = get_act_code(t_now)
        
        # Пытаемся заглянуть в "прошлое" и "будущее" (сканируем волновую функцию)
        # Если код на другом устройстве совпадет с одним из этих, значит мы нашли дрейф!
        print(f"TIME: {t_now} | LOCAL CODE: {local_code}")
        
        # Выводим подсказку: какие коды сейчас в "соседних" реальностях
        past_code = get_act_code(t_now - 100)
        future_code = get_act_code(t_now + 100)
        print(f"Scan Window: [ {past_code} <--- {local_code} ---> {future_code} ]")
        
        time.sleep(1)
except KeyboardInterrupt:
    pass
