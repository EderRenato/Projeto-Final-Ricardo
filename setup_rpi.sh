#!/bin/bash
# ============================================================
# Setup do Raspberry Pi Zero 2 W para Edge ML
# Executar: chmod +x setup_rpi.sh && ./setup_rpi.sh
# ============================================================
echo "=========================================="
echo " Setup — Edge ML (RPi Zero 2 W)"
echo "=========================================="

echo ""
echo "[1/3] Atualizando sistema..."
sudo apt-get update -y && sudo apt-get upgrade -y

echo ""
echo "[2/3] Instalando dependências..."
sudo apt-get install -y python3-pip python3-numpy libatlas-base-dev

echo ""
echo "[3/3] Instalando tflite-runtime..."
pip3 install tflite-runtime --break-system-packages

echo ""
echo "Verificando instalação..."
python3 -c "
import numpy as np
print(f'  NumPy: {np.__version__}')
try:
    from tflite_runtime.interpreter import Interpreter
    print('  tflite-runtime: OK ✅')
except ImportError:
    print('  tflite-runtime: FALHOU ❌')
"

mkdir -p ~/edge_ml
echo ""
echo "=========================================="
echo " ✅ Setup concluído!"
echo ""
echo " Copie os arquivos para ~/edge_ml/:"
echo "   scp exports/* pi@<IP>:~/edge_ml/"
echo ""
echo " Execute:"
echo "   cd ~/edge_ml"
echo "   python3 inference.py"
echo "=========================================="
