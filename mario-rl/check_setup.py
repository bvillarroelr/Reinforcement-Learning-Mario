# check_setup.py
import time
from src.env_setup import make_mario_env

def main():
    print("1. Cargando entorno con Wrappers...")
    try:
        env = make_mario_env()
        print("   -> ¡Entorno cargado correctamente!")
    except Exception as e:
        print(f"   -> ERROR FATAL cargando el entorno: {e}")
        return

    print("2. Reseteando el juego...")
    obs = env.reset()
    
    # Verificación técnica: ¿Qué tamaño de imagen recibe la IA?
    # Debería ser (1, 84, 84, 4) o (4, 84, 84) dependiendo de la config
    print(f"   -> La IA está viendo una imagen de tamaño: {obs.shape}")

    print("3. Iniciando prueba visual (Presiona Ctrl+C en la terminal para salir)...")
    print("   Nota: Verás a Mario moverse aleatoriamente.")
    
    try:
        # Loop infinito hasta que tú lo pares
        while True:
            # Acción aleatoria
            action = [env.action_space.sample()]
            
            # Paso del juego
            obs, rewards, dones, info = env.step(action)
            
            # Renderizar en pantalla
            env.render()
            
            # Dormir un poco para que el ojo humano pueda verlo (sino va muy rápido)
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nPrueba finalizada por el usuario.")
    finally:
        env.close()
        print("Juego cerrado.")

if __name__ == "__main__":
    main()