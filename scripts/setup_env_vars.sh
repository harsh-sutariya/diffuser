#!/bin/bash

# =============================================================================
# Diffuser Environment Setup Script
# =============================================================================
# This script sets up all necessary environment variables to run diffusion 
# model training in containerized/headless environments without graphics issues.
#
# Usage:
#   source scripts/setup_env_vars.sh    # To set vars in current shell
#   ./scripts/setup_env_vars.sh         # To run as standalone script
# =============================================================================

echo "🚀 Setting up Diffuser environment variables..."

# =============================================================================
# Graphics and Rendering Fixes
# =============================================================================
echo "📺 Configuring graphics/rendering settings..."

# Disable EGL/OpenGL rendering to avoid GLIBC issues in containers
export EGL_PLATFORM=surfaceless
export PYOPENGL_PLATFORM=osmesa

# Force Mesa software rendering (avoid hardware dependencies)
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_GLSL_VERSION_OVERRIDE=330

# Set MuJoCo to use software rendering
export MUJOCO_GL=osmesa

# Disable display for headless operation
export DISPLAY=""
unset DISPLAY

echo "✅ Graphics settings configured for headless operation"

# =============================================================================
# D4RL Configuration
# =============================================================================
echo "🎮 Configuring D4RL environment..."

# Suppress D4RL import errors for missing optional dependencies
export D4RL_SUPPRESS_IMPORT_ERROR=1

echo "✅ D4RL configured to suppress import warnings"

# Disable TensorFlow warnings if installed
export TF_CPP_MIN_LOG_LEVEL=2

# =============================================================================
# Verification and Status
# =============================================================================

echo ""
echo "🌟 Key Environment Variables Set:"
echo "=================================="
echo "D4RL_SUPPRESS_IMPORT_ERROR=$D4RL_SUPPRESS_IMPORT_ERROR"
echo "MUJOCO_GL=$MUJOCO_GL"
echo "PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM"
echo "EGL_PLATFORM=$EGL_PLATFORM"
echo "MESA_GL_VERSION_OVERRIDE=$MESA_GL_VERSION_OVERRIDE"

echo ""
echo "Export setup complete!"
echo ""