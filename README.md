# Edge ML: Classificação de Falhas em Equipamentos Rotativos

Este projeto implementa uma pipeline completa de Inteligência Artificial para a detecção de falhas em máquinas industriais, otimizada para execução em hardware limitado (**Raspberry Pi Zero 2 W**). O sistema utiliza processamento de sinais de vibração, acústica e temperatura para classificar o estado do equipamento. 

## 🛠️ Tecnologias e Ferramentas

* 
**Linguagem:** Python 3.10 


* 
**Modelagem:** TensorFlow/Keras (MLP), Random Forest, XGBoost 


* 
**Edge Computing:** TFLite Runtime (Modelos Float32 e INT8 Quantizado) 


* 
**Geração de Dados:** Conditional GAN (cGAN) para aumento de dados sintéticos 


* 
**Hardware Alvo:** Raspberry Pi Zero 2 W 



## 📊 Arquitetura do Modelo (MLP)

O modelo principal é uma rede neural perceptron multicamadas (MLP) com a seguinte estrutura: 

1. 
**Camada de Entrada:** 5 features (Vibração X, Y, Z, Nível Acústico e Temperatura) 


2. 
**Dense (64 unidades)** + Batch Normalization + Dropout (0.3) 


3. 
**Dense (32 unidades)** + Batch Normalization + Dropout (0.2) 


4. 
**Dense (16 unidades)** 


5. 
**Saída (Softmax):** 4 classes de falha 



### Classes Monitoradas:

* 
`0`: Bearing Fault (Falha de Rolamento) 


* 
`1`: Imbalance (Desbalanceamento) 


* 
`2`: Normal (Operação Normal) 


* 
`3`: Overheating (Superaquecimento) 



## 🚀 Como Executar no Raspberry Pi

### 1. Preparação do Ambiente

Utilize o script de setup fornecido para instalar as dependências necessárias (NumPy, Libatlas e TFLite Runtime): 

```bash
chmod +x setup_rpi.sh
./setup_rpi.sh

```

### 2. Inferência e Benchmark

O script `inference1.py` permite rodar o modelo em diferentes modos: 

```bash
# Executar teste completo (Dados reais + Sintéticos + Benchmark)
python3 inference1.py --mode all

# Apenas benchmark de latência no hardware
python3 inference1.py --mode benchmark --model model_quantized_int8.tflite

```

## 📈 Resultados e Visualizações

O projeto gera uma série de análises detalhadas:

* 
**Análise Exploratória:** Distribuição de features (`02_feature_distributions.png`) e correlação (`03_correlation.png`). 


* 
**Desempenho:** Curvas de aprendizado (`07_training_curves.png`) e validação K-Fold (`08_kfold_results.png`). 


* 
**Métricas:** Matriz de Confusão (`10_confusion_matrix.png`) e Curva ROC Multiclasse (`11_roc_curve.png`). 


* 
**Dados Sintéticos:** Perda da GAN (`13_gan_loss.png`) e comparação PCA entre dados reais e sintéticos (`14_pca_real_vs_synthetic.png`). 



## 📂 Estrutura de Arquivos Principal

* 
`inference1.py`: Script principal de inferência otimizada. 


* 
`model_quantized_int8.tflite`: Modelo ultra-leve para o Pi Zero. 


* 
`scaler_params.json`: Parâmetros de normalização (Z-score). 


* 
`test_data.csv`: Dataset de 15% reservado para teste final (evitando *data leakage*). 




## 👥 Autores

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/EderRenato">
        <img src="https://github.com/EderRenato.png" width="100px;" alt="Eder Renato"/><br>
        <sub><b>Eder Renato</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Beroradin">
        <img src="https://github.com/Beroradin.png" width="100px;" alt="Matheus Pereira"/><br>
        <sub><b>Matheus Pereira</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Brunis1108">
        <img src="https://github.com/Brunis1108.png" width="100px;" alt="Bruna Alves"/><br>
        <sub><b>Bruna Alves</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/marifariasz">
        <img src="https://github.com/marifariasz.png" width="100px;" alt="Mariana Silva"/><br>
        <sub><b>Mariana Silva</b></sub>
      </a>
    </td>
  </tr>
</table>

