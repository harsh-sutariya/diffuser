from .serialization import *
from .training import *
from .progress import *
from .setup import *
from .config import *
#from .rendering import *
from .arrays import *
from .colab import *
from .logger import *

# Dummy renderer for headless training (when mujoco_py is not available)
class DummyRenderer:
    """Stub renderer that does nothing - for headless training"""
    
    def __init__(self, env):
        print("[ utils ] Using DummyRenderer (no visualization)")
        self.env = env if isinstance(env, str) else str(env)
        # Set dimensions for compatibility
        self.observation_dim = 17  # typical for locomotion tasks
        self.action_dim = 6       # typical for locomotion tasks
    
    def composite(self, savepath, observations, **kwargs):
        """Stub method - does nothing"""
        if savepath:
            print(f"[ DummyRenderer ] Would save visualization to: {savepath}")
    
    def render_plan(self, savepath, actions, observations, state, **kwargs):
        """Stub method - does nothing"""
        if savepath:
            print(f"[ DummyRenderer ] Would save plan video to: {savepath}")
    
    def render_rollout(self, savepath, states, **kwargs):
        """Stub method - does nothing"""
        if savepath:
            print(f"[ DummyRenderer ] Would save rollout video to: {savepath}")
    
    def render(self, observation, **kwargs):
        """Stub method - returns black image"""
        return None
    
    def __call__(self, *args, **kwargs):
        return None

# Use DummyRenderer as MuJoCoRenderer for compatibility
MuJoCoRenderer = DummyRenderer
