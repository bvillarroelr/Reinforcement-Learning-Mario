import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from src.env_setup import make_mario_env

# --- CONFIGURACIÓN MATEMÁTICA ---
PASO_INICIAL = 550000      # Donde estás ahora
META_FINAL = 1000000       # Donde quieres llegar
PASOS_A_ENTRENAR = META_FINAL - PASO_INICIAL # 450,000 pasos
# --------------------------------

class TrainAndRenderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainAndRenderCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        self.training_env.render()
        return True

def main():
    # 1. Rutas a prueba de balas (Absolutas)
    directorio_base = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINT_DIR = os.path.join(directorio_base, 'train', 'checkpoints')
    LOG_DIR = os.path.join(directorio_base, 'train', 'logs')
    
    # 2. Construir nombre del archivo a cargar
    # El código anterior guardó con prefijo 'mario_finetune', así que buscamos ese.
    nombre_archivo = f"mario_finetune_{PASO_INICIAL}_steps"
    ruta_modelo = os.path.join(CHECKPOINT_DIR, nombre_archivo)

    print(f"--> Buscando checkpoint de la Etapa 2: {ruta_modelo}")

    env = make_mario_env()

    try:
        # 3. CARGAR MODELO
        model = PPO.load(
            ruta_modelo, 
            env=env, 
            tensorboard_log=LOG_DIR,
            # Mantenemos los hiperparámetros de Fine-Tuning
            custom_objects={'learning_rate': 0.00001, 'clip_range': 0.1}
        )
        print("--> ¡Modelo cargado! Iniciando la recta final hacia el Millón.")
    except FileNotFoundError:
        print(f"\n[ERROR] No existe el archivo: {ruta_modelo}.zip")
        print(f"Por favor verifica en la carpeta {CHECKPOINT_DIR} cuál es el nombre exacto del último archivo zip.")
        return

    # 4. Callbacks
    # Cambiamos el prefijo para identificar los archivos de esta etapa final
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=CHECKPOINT_DIR,
        name_prefix='mario_million_run' 
    )
    
    render_callback = TrainAndRenderCallback()

    print("--------------------------------------------------")
    print(f" INICIANDO ETAPA FINAL: {PASOS_A_ENTRENAR} PASOS")
    print(f" Meta: Llegar al paso {META_FINAL}")
    print("--------------------------------------------------")
    
    try:
        model.learn(
            total_timesteps=PASOS_A_ENTRENAR, # 450,000
            callback=[checkpoint_callback, render_callback],
            reset_num_timesteps=False, # CLAVE: Para que TensorBoard siga sumando
            tb_log_name="PPO_final_stretch" # Nombre nuevo para diferenciarlo en la gráfica
        )
        
    except KeyboardInterrupt:
        print("\nDetenido manualmente.")
    
    # Guardado final definitivo
    model.save("MARIO_CAMPEON_1M")
    print("¡ENTRENAMIENTO COMPLETADO! 1 MILLÓN DE PASOS.")
    env.close()

if __name__ == "__main__":
    main()