import os
from xml.parsers.expat import model
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from src.env_setup import make_mario_env

# --- CONFIGURACIÓN ---
PASO_A_CARGAR = 50000  # Tu checkpoint seguro
# ---------------------

class TrainAndRenderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainAndRenderCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        self.training_env.render()
        return True

def main():
    CHECKPOINT_DIR = './train/checkpoints/'
    LOG_DIR = './train/logs/'  # <--- Ahora sí la usaremos
    
    # Construir nombre del archivo
    nombre_archivo = f"mario_model_{PASO_A_CARGAR}_steps"
    ruta_modelo = os.path.join(CHECKPOINT_DIR, nombre_archivo)

    print(f"--> Buscando checkpoint: {ruta_modelo}")

    env = make_mario_env()

    try:
        # CARGAR MODELO
        model = PPO.load(
            ruta_modelo, 
            env=env, 
            tensorboard_log=LOG_DIR, # Le decimos dónde guardar las nuevas gráficas
            custom_objects={'learning_rate': 0.00001, 'clip_range': 0.1}
        )
        print("--> ¡Modelo cargado correctamente con Learning Rate bajo!")
    except FileNotFoundError:
        print(f"\n[ERROR] No existe el archivo: {ruta_modelo}.zip")
        return

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=CHECKPOINT_DIR,
        name_prefix='mario_finetune' 
    )
    
    render_callback = TrainAndRenderCallback()

    print("--------------------------------------------------")
    print(f" FINE-TUNING DESDE PASO {PASO_A_CARGAR}")
    print(" Se creará una nueva carpeta en logs (ej. PPO_4)")
    print("--------------------------------------------------")
    
    try:
        model.learn(
            total_timesteps=500000, 
            callback=[checkpoint_callback, render_callback],
            reset_num_timesteps=False,
            tb_log_name="PPO_fine_tune" 
        )

    except KeyboardInterrupt:
        print("\nDetenido manualmente.")
    
    model.save("mario_final_finetuned")
    env.close()

if __name__ == "__main__":
    main()