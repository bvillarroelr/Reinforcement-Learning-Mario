import os
import time
from stable_baselines3 import PPO
# Importamos tu configuración exacta de entorno
from src.env_setup import make_mario_env

# ==============================================================================
#CONFIGURACIÓN
# ==============================================================================

NOMBRE_ARCHIVO = "mario_finetune_150000_steps"

# ==============================================================================

def main():
    # 1. Configuración de Rutas "Inteligente"
    # Esto encuentra la carpeta donde está este script, sin importar tu usuario
    directorio_base = os.path.dirname(os.path.abspath(__file__))
    directorio_checkpoints = os.path.join(directorio_base, "train", "checkpoints")
    
    # Construimos la ruta completa
    ruta_modelo = os.path.join(directorio_checkpoints, NOMBRE_ARCHIVO)

    print("----------------------------------------------------------")
    print(f"--> Buscando al candidato: {NOMBRE_ARCHIVO}")
    print("----------------------------------------------------------")

    # 2. Verificar si existe el archivo (con o sin .zip)
    if not os.path.exists(ruta_modelo + ".zip"):
        print(f"\n[ERROR CRÍTICO] No encuentro el archivo en:")
        print(f"{ruta_modelo}.zip")
        print("\nCONSEJOS:")
        print("1. Ve a la carpeta 'train/checkpoints'.")
        print("2. Copia el nombre exacto del archivo (F2 -> Ctrl+C).")
        print("3. Pégalo en la variable NOMBRE_ARCHIVO de este script.")
        return

    # 3. Cargar el Entorno
    print("--> Creando entorno de Mario...")
    env = make_mario_env()
    env.reset()

    # 4. Cargar el Cerebro (Modelo)
    print(f"--> Cargando cerebro...")
    try:
        # custom_objects es necesario por si cambiaste el learning_rate en el fine-tuning
        # Esto evita errores de carga si los parámetros no coinciden exacto
        model = PPO.load(ruta_modelo, env=env, custom_objects={'learning_rate': 0.00001, 'clip_range': 0.1})
    except Exception as e:
        print(f"\n[ERROR] Falló la carga del modelo. Detalle: {e}")
        # Intento de rescate: Cargar sin custom_objects por si es un modelo viejo
        print("Intento 2: Cargando sin parámetros personalizados...")
        model = PPO.load(ruta_modelo, env=env)

    print("\n--> ¡ÉXITO! Ventana de juego activa.")
    print("--> Presiona Ctrl+C en esta terminal para detener/probar otro.")

    # 5. Bucle de Juego
    obs = env.reset()
    try:
        while True:
            # La IA decide qué hacer
            action, _ = model.predict(obs)
            
            # Ejecutar acción
            obs, reward, done, info = env.step(action)
            
            # Mostrar pantalla
            env.render()
            
            # Control de velocidad (para verlo como humano)
            time.sleep(0.0006) # ~60 FPS
            
    except KeyboardInterrupt:
        print("\nPrueba finalizada por el usuario.")
    finally:
        env.close()
        print("Ventana cerrada.")

if __name__ == "__main__":
    main()