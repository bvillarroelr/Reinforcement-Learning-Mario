import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from src.env_setup import make_mario_env

# --- PARTE NUEVA: EL VISUALIZADOR ---
class TrainAndRenderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainAndRenderCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Esto accede al entorno y le dice "muéstrate"
        self.training_env.render()
        return True

def main():
    CHECKPOINT_DIR = './train/checkpoints/'
    LOG_DIR = './train/logs/'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("Iniciando entorno de Mario...")
    env = make_mario_env()

    model = PPO(
        policy='CnnPolicy',
        env=env,
        learning_rate=0.0001,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=LOG_DIR,
        verbose=1
    )

    # Callback 1: Guardar backup
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=CHECKPOINT_DIR,
        name_prefix='mario_model'
    )
    
    # Callback 2: Renderizar (Ver el juego)
    render_callback = TrainAndRenderCallback()

    print("--------------------------------------------------")
    print(" MODO VISUAL ACTIVADO:")
    print("--------------------------------------------------")
    
    # Pasamos una LISTA de callbacks [guardar, ver]
    try:
        model.learn(total_timesteps=1000000, callback=[checkpoint_callback, render_callback])
    except KeyboardInterrupt:
        print("\nEntrenamiento detenido manualmente.")
    
    model.save("mario_final_best")
    env.close()

if __name__ == "__main__":
    main()