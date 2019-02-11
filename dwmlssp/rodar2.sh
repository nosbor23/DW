#!/bin/bash
echo "Inicio dos testes"

python dwp1.py

echo "Testes finalizados, compactando..."
cp log0802.txt ~/Thiago/Robson/DW/dwmlssp/result/
python zipador.py
echo "compactado, enviando e-mail..."
python semail.py
echo "Finalizado"
echo

echo "Inicio dos testes"

python dwp2.py

echo "Testes finalizados, compactando..."
cp log0802.txt ~/Thiago/Robson/DW/dwmlssp/result/
python zipador.py
echo "compactado, enviando e-mail..."
python semail.py
echo "Finalizado"
echo

echo "Inicio dos testes"

python dwp3.py

echo "Testes finalizados, compactando..."
cp log0802.txt ~/Thiago/Robson/DW/dwmlssp/result/
python zipador.py
echo "compactado, enviando e-mail..."
python semail.py
echo "Finalizado"
echo

echo "Inicio dos testes"

python dwp4.py

echo "Testes finalizados, compactando..."
cp log0802.txt ~/Thiago/Robson/DW/dwmlssp/result/
python zipador.py
echo "compactado, enviando e-mail..."
python semail.py
echo "Finalizado"
echo


