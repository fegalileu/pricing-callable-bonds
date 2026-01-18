# Framework de Precificação de Títulos Resgatáveis: Uma Abordagem Comparativa

![Language](https://img.shields.io/badge/python-3.9%2B-blue?style=for-the-badge&logo=python)
![Library](https://img.shields.io/badge/QuantLib-1.30%2B-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-MBA_Eng_Financeira-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)

##  Objetivo

O objetivo central da pesquisa é isolar e quantificar o **Risco de Modelo** contrastando abordagens de **Não-Arbitragem** (Hull-White, Black-Karasinski) e **Equilíbrio** (Cox-Ingersoll-Ross). A metodologia privilegia a implementação manual (*from scratch*) dos motores de cálculo (EDP e Árvores), utilizando a biblioteca de mercado *QuantLib* apenas como *benchmark* de verificação.

---

## 🚀 Reprodutibilidade e Uso

Para garantir a reprodutibilidade dos resultados apresentados na monografia, siga as instruções abaixo.

### Pré-requisitos
* Python 3.9 ou superior
* Gerenciador de pacotes `pip`

### Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/callable-bond-pricing.git
cd callable-bond-pricing

# 2. Instale as dependências listadas
pip install -r requirements.txt
```

### Execução da Análise
O script principal orquestra a ingestão de dados, calibração dos modelos e geração das tabelas de resultados.

```bash
python run_analysis.py
```

Os artefatos gerados (csv, plots) serão salvos na pasta `outputs/`.

---

## 🧠 Modelagem e Algoritmos

Três paradigmas de modelagem foram implementados e confrontados:

### 1. Modelo Hull-White (HW) - 1 Fator
* **Dinâmica:** Gaussiana com Reversão à Média Time-Dependent.
  $$dr_t = [\theta(t) - a r_t]dt + \sigma dW_t$$
* **Implementação:** Simulação de Monte Carlo com regressão de Mínimos Quadrados (LSMC) para a fronteira de exercício ótimo (Bermudan/American).
* **Técnica:** Uso de *Common Random Numbers (CRN)* para cálculo estável de Gregas (Duration/Convexity).

### 2. Modelo Black-Karasinski (BK)
* **Dinâmica:** Log-normal na taxa curta (garante $r_t > 0$).
  $$d(\ln r_t) = [\theta(t) - a \ln r_t]dt + \sigma dW_t$$
* **Implementação:** Árvore Trinomial Recombinante.
* **Técnica:** Calibração exata via *Forward Induction* no termo de drift $\theta(t)$ para recuperar a estrutura a termo inicial.

### 3. Modelo Cox-Ingersoll-Ross (CIR)
* **Dinâmica:** Difusão de Raiz Quadrada (Feller condition).
  $$dr_t = \kappa(\theta - r_t)dt + \sigma \sqrt{r_t} dW_t$$
* **Implementação:** Método de Diferenças Finitas (FDM) implícito (Crank-Nicolson) para solução da EDP de precificação.
* **Técnica:** Condições de contorno reflexivas em $r=0$ e lineares assintóticas para grandes taxas.

---

## 📊 Principais Resultados

A tabela a seguir apresenta os resultados de precificação para um título comparável ao **Microsoft Corp. Callable 2035**, calibrado com curva SOFR e spread de crédito (OAS) de 75bps.

| Modelo / Método Numérico | Preço ($) | Duration | Convexidade | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Straight Bond (Benchmark)** | **92.36** | **8.70** | **84.61** | *Valor Teórico S/ Opção* |
| Hull-White (LSMC Manual) | 91.63 | 8.35 | 77.81 | ✅ Validado |
| Hull-White (QuantLib Tree) | 92.03 | 8.47 | 80.21 | ✅ Validado |
| Black-Karasinski (Tree Manual) | 91.30 | 8.48 | 82.63 | ✅ Validado |
| **CIR (PDE Manual)** | **97.75** | **7.85** | **64.67** | ⚠️ **Divergência Esperada** |

### Discussão sobre o Modelo CIR
A discrepância observada no modelo CIR (**97.75** vs **~91.60**) ilustra o **Risco de Modelo**. O CIR, sendo um modelo de equilíbrio, força a reversão da taxa para uma média histórica de longo prazo ($\theta$). Em cenários onde a curva de juros futura (Forward) está precificando taxas muito acima dessa média histórica, o modelo subestima as taxas de desconto, superavaliando o preço do título. Isso confirma a inadequação de modelos de equilíbrio puro para *pricing* ativo sem a extensão de deslocamento determinístico (Ex-CIR).

---

## 📚 Referências Bibliográficas

As implementações baseiam-se nos trabalhos seminais da literatura de derivativos de taxas de juros:

1. **Hull, J., & White, A. (1990).** Pricing Interest-Rate-Derivative Securities. *The Review of Financial Studies*, 3(4), 573–592.
2. **Black, F., & Karasinski, P. (1991).** Bond and Option Pricing when Short Rates are Lognormal. *Financial Analysts Journal*, 47(4), 52–59.
3. **Cox, J. C., Ingersoll, J. E., & Ross, S. A. (1985).** A Theory of the Term Structure of Interest Rates. *Econometrica*, 53(2), 385–407.
4. **Andersen, L. B. G. (2000).** A Simple Approach to the Pricing of Bermudan Swaptions in the Multi-Factor LIBOR Market Model. *Journal of Computational Finance*.

---

## 📝 Citação

Caso utilize este código ou os resultados em trabalhos acadêmicos, por favor cite:

```bibtex
@monograph{CallableBondPricing2026,
  author  = {GALILEU, Felipe},
  title   = {Precificação de títulos corporativos resgatáveis (callable): comparação entre árvore recombinante, EDP e simulação de Monte Carlo sob modelos unifatoriais de taxa curta},
  school  = {Escola Politécnica da Universidade de São Paulo},
  year    = {2026},
  type    = {Monografia de Pós-Graduação MBA}
}
```

---
*Desenvolvido no contexto de pesquisa acadêmica. Não constitui recomendação de investimento.*