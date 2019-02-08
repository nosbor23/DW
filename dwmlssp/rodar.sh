#!/bin/bash
echo "Inicio dos testes"

python dwp.py

echo "Testes finalizados, compactando..."
python zipador.py
echo "compactado, enviando e-mail..."
python semail.py
echo "Finalizado"
echo


