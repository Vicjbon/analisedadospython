# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 09:04:02 2026

@author: User
"""

### PROJETO ONLINE RETAIL #### MODELAGEM MULTINIVEL
## PERGUNTA 1 -O que gera mais faturamento: clientes que compram barato, mas sempre, 
# ou clientes que compram caro, mas raramente?

### importar bibliotecas ####
!pip install openpyxl

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import mixedlm
from scipy.stats import chi2
import time

##### Carregar e preparar dados #####

df = pd.read_excel('online_retail.xlsx')
print("Primeiras 5 linhas:")
print(df.head())

## converter para CSV###
df.to_csv('online_retail.xlsx', index=False)
print("Convertido para CSV!")

# Remover CustomerID nulos
df = df.dropna(subset=['Customer ID'])
df['Customer ID'] = df['Customer ID'].astype(int).astype(str)

print(df.dtypes)
df.info()

# Criar valor total do item
df['ItemAmount'] = df['Quantity'] * df['Price']

print(f"✓ Após limpeza: {len(df):,} linhas")
print(f"✓ Clientes únicos: {df['Customer ID'].nunique():,}")
print(f"✓ Faturas únicas: {df['Invoice'].nunique():,}")

## Renomear a coluna ####
df = df.rename(columns={'Customer ID': 'CustomerID'})

##### Agregar o nivel 2####### FATURA
print("\n[3/8] Agregando para nível 2 (faturas)...")

df_invoice = df.groupby(['Invoice', 'CustomerID', 'InvoiceDate', 'Country']).agg(
    TotalAmount=('ItemAmount', 'sum'),
    NumItems=('StockCode', 'count')
).reset_index()

print(f"✓ {len(df_invoice):,} faturas criadas")

customer_metrics = df.groupby('CustomerID').agg(
    Frequency=('Invoice', 'nunique'),           # Número de faturas
    CustomerAvgPrice=('Price', 'mean'),         # Preço médio dos itens
    TotalCustomerRevenue=('ItemAmount', 'sum')  # Faturamento total
).reset_index()

# Juntar nível 3 ao nível 2
df_invoice = df_invoice.merge(customer_metrics, on='CustomerID', how='left')

print(f"✓ {len(customer_metrics):,} clientes")

### Tratar os valores negativos #####
# Mostrar clientes com valores negativos
neg_revenue = customer_metrics[customer_metrics['TotalCustomerRevenue'] < 0]
neg_revenue

customer_metrics_clean = customer_metrics[
    (customer_metrics['Frequency'] >= 0) &
    (customer_metrics['CustomerAvgPrice'] >= 0) &
    (customer_metrics['TotalCustomerRevenue'] >= 0)
]

# Juntar nível 3 ao nível 2
df_invoice = df_invoice.merge(customer_metrics_clean, on='CustomerID', how='left')
df_invoice

df_invoice.columns

### remover algumas colunas ####
df_invoice = df_invoice.drop(columns=[
    'Frequency_x', 'CustomerAvgPrice_x', 'TotalCustomerRevenue_x',
    'Frequency_y', 'CustomerAvgPrice_y', 'TotalCustomerRevenue_y'
])

## contagem de valores faltantes###
df_invoice.isna().sum()

## removendo linhas com valores faltantes ###
df_invoice = df_invoice.dropna()

### remover os negativos de total amount####
df_invoice = df_invoice[df_invoice['TotalAmount'] > 0]

# Transformar para log #### Estabiliza a variancia ####
df_invoice['logTotalAmount'] = np.log1p(df_invoice['TotalAmount'])
df_invoice['logFrequency'] = np.log1p(df_invoice['Frequency'])
df_invoice['logCustomerAvgPrice'] = np.log1p(df_invoice['CustomerAvgPrice'])

#### dados organizados e tratados, agora iniciar o trabalho de analises ###
##### ANALISE DESCRITIVA DOS DADOS ######

# 1.2 Distribuição dos clientes por frequência
print("\n1.2 Distribuição da frequência de compras:")
print(df_invoice['Frequency'].value_counts().sort_index().head(50))


print("\n1.3 Percentis do TotalAmount por fatura:")
percentis = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
for p in percentis:
    valor = df_invoice['TotalAmount'].quantile(p)
    print(f"  Percentil {p*100:5.1f}%: R$ {valor:.2f}")

# Quantas faturas por cliente?
faturas_por_cliente = df_invoice.groupby('CustomerID').size()
print("\n2.1 Distribuição de faturas por cliente:")
print(f"  Mínimo: {faturas_por_cliente.min()}")
print(f"  Máximo: {faturas_por_cliente.max()}")
print(f"  Média: {faturas_por_cliente.mean():.2f}")
print(f"  Mediana: {faturas_por_cliente.median():.2f}")

#### Analise gráfica exploratória ###### CORRIGIR ESSA PARTE #####
########
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.figure(figsize=(10, 6))

# Histograma faturamento por fatura (apenas valores até o percentil 95 para melhor visualização) remove os valores mais altos (5%) - Outliers
limite = df_invoice['TotalAmount'].quantile(0.95)
dados_filtrados = df_invoice[df_invoice['TotalAmount'] <= limite]

sns.histplot(data=dados_filtrados, x='TotalAmount', bins=50, edgecolor='gray', color='steelblue', alpha=0.7)
plt.xlabel('Total da fatura ($)', fontsize=12)
plt.ylabel('Número de faturas', fontsize=12)
plt.title('Distribuição do Faturamento por Fatura', fontsize=14)
plt.grid(False)

# Adicionar linha da média
media = df_invoice['TotalAmount'].mean()
plt.axvline(media, color='black', linestyle='--', linewidth=1, label=f'Média: $ {media:.2f}')

# Adicionar linha da mediana
mediana = df_invoice['TotalAmount'].median()
plt.axvline(mediana, color='darkblue', linestyle='--', linewidth=1, label=f'Mediana: $ {mediana:.2f}')

plt.legend()
plt.show()

##### distribuição de vendas/faturamento - usando log ##### Observa-se uma curva NORMAL
df_invoice["log_TotalAmount"] = np.log1p(df_invoice["TotalAmount"])

sns.histplot(df_invoice["logTotalAmount"], bins=30, kde=True, color='salmon')
plt.title("Log-Transformed Purchase Distribution")
plt.show()

##### Distribuição das VENDAS POR PAIS - DEZ PRIMEIROS ####
top_countries = df_invoice["Country"].value_counts().head(10).index

sns.boxplot(
    data=df_invoice[df_invoice["Country"].isin(top_countries)], color ="darkred",
    x="Country",
    y="TotalAmount"
)
plt.xticks(rotation=45)
plt.title('Purchase Value by Country')
plt.show()

##### Faturamento total por Pais#####
df_country = (df_invoice.groupby("Country")['TotalAmount'].sum().sort_values(ascending=False))
plt.figure(figsize=(10,5))

sns.barplot(x=df_country.index[:10],y=df_country.values[:10],color="black")
plt.xticks(rotation=45)
plt.xlabel("País")
plt.ylabel("Faturamento total ($)")
plt.title("Faturamento total por país (Top 10)")

plt.show()
    
##### CORRELAÇÃO entre variáveis ######
corr = df_invoice.select_dtypes(include=np.number).corr()
sns.heatmap(corr, annot=True, cmap="viridis")
plt.title("Correlação entre variáveis")

#######
sns.scatterplot(
    data=df_invoice,
    x="Frequency",
    y="TotalCustomerRevenue"
)

#### Distribuição da frequecnia #######
sns.histplot(df_invoice["Frequency"], bins=100, kde=True, color ="black")
plt.title("Customer Purchase Frequency Distribution")
plt.show()
######
#######

print("\n✅ GRÁFICO 1 CONCLUÍDO")

###FREQUÊNCIA vs GASTO MÉDIO
# ============================================

import matplotlib.pyplot as plt

# Calcular gasto médio por nível de frequência
avg_by_freq = df_invoice.groupby('Frequency')['TotalAmount'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.plot(avg_by_freq['Frequency'], avg_by_freq['TotalAmount'], 'o-', linewidth=2, markersize=6, color='coral')
plt.xlabel('Número de compras (frequência do cliente)', fontsize=12)
plt.ylabel('Gasto médio por fatura (R$)', fontsize=12)
plt.title('Relação: Clientes mais frequentes gastam mais por fatura?', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()


###### Grupos de preço###
# Criar grupos de preço
df_invoice['PriceGroup'] = pd.qcut(df_invoice['CustomerAvgPrice'], q=4, labels=['Mais Baratos', 'Baratos', 'Caros', 'Mais Caros'])

# Calcular faturamento médio por grupo
price_performance = df_invoice.groupby('PriceGroup')['TotalAmount'].mean().reindex(['Mais Baratos', 'Baratos', 'Caros', 'Mais Caros'])

plt.figure(figsize=(8, 6))
bars = plt.bar(price_performance.index, price_performance.values, color=['skyblue', 'steelblue', 'coral', 'red'], edgecolor='black')
plt.xlabel('Perfil do cliente (preço médio dos itens)', fontsize=12)
plt.ylabel('Gasto médio por fatura (R$)', fontsize=12)
plt.title('Clientes que compram itens mais caros gastam mais?', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')

# Adicionar valores nas barras
for bar, valor in zip(bars, price_performance.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f'R$ {valor:.0f}', ha='center', fontsize=10)

plt.show()




##### MODELO NULO #####

from statsmodels.formula.api import mixedlm

# Modelo sem nenhum preditor (apenas intercepto)
model_null = mixedlm(
    "logTotalAmount ~ 1",
    df_invoice,
    groups=df_invoice["CustomerID"],
    re_formula = "~1"
).fit()

print(model_null.summary())

var_between = model_null.cov_re.iloc[0, 0]  # variância entre clientes
var_within = model_null.scale               # variância dentro do cliente
icc = var_between / (var_between + var_within)

print(f"\n📊 INTERPRETAÇÃO DO MODELO NULO:")
print(f"  Variância ENTRE clientes (τ₀²): {var_between:.4f}")
print(f"  Variância DENTRO do cliente (σ²): {var_within:.4f}")
print(f"  ICC = {icc:.4f}")
print(f"\n  → {icc*100:.1f}% da variância no faturamento se deve a diferenças ENTRE clientes")
print(f"  → {100-icc*100:.1f}% da variância se deve a diferenças DENTRO do mesmo cliente")

#### Modelo com efeitos fixos - Sem interação #####
model_fixed = mixedlm(
    "logTotalAmount ~ NumItems + logFrequency + logCustomerAvgPrice",
    df_invoice,
    groups=df_invoice["CustomerID"],
    re_formula="~1"
).fit()

print(model_fixed.summary())

print(f"    → Cada item adicional na fatura aumenta o faturamento em X%")
print(f"  • logFrequency (γ₀₁): {model_fixed.params['logFrequency']:.4f}")
print(f"    → Clientes com maior frequência gastam Y% mais por fatura")
print(f"  • logCustomerAvgPrice (γ₀₂): {model_fixed.params['logCustomerAvgPrice']:.4f}")
print(f"    → Clientes que compram itens mais caros gastam Z% mais por fatura")

#### Modelo com interação #####

model_interaction = mixedlm(
    "logTotalAmount ~ NumItems + logFrequency * logCustomerAvgPrice",
    df_invoice,
    groups=df_invoice["CustomerID"],
    re_formula="~1"
).fit()

print(model_interaction.summary())

########

# Extrair coeficientes
coef_freq = model_interaction.params['logFrequency']
coef_price = model_interaction.params['logCustomerAvgPrice']
coef_interaction = model_interaction.params['logFrequency:logCustomerAvgPrice']
p_interaction = model_interaction.pvalues['logFrequency:logCustomerAvgPrice']

print("\n" + "="*60)
print("RESPOSTA PARA SUA PERGUNTA")
print("="*60)
print(f"Coeficiente de interação: {coef_interaction:.4f}")
print(f"P-valor: {p_interaction:.6f}")

if p_interaction < 0.05:
    if coef_interaction < 0:
        print("\n✅ CONCLUSÃO: ALTA FREQUÊNCIA + ITENS BARATOS")
        print("   O maior faturamento vem de clientes que compram")
        print("   com frequência itens de baixo preço.")
    else:
        print("\n✅ CONCLUSÃO: BAIXA FREQUÊNCIA + ITENS CAROS")
        print("   O maior faturamento vem de clientes que compram")
        print("   itens caros, mesmo que com menor frequência.")
else:
    print("\n⚠️ Interação NÃO significativa.")
    print("   Frequência e preço afetam o faturamento de forma independente.")

### RESPOSTA >> BAIXA FREQUÊNCIA + ITENS CAROS gera MAIOR FATURAMENTO. #####

###### Comparação dos modelos #########

from scipy.stats import chi2

print("\n" + "="*60)
print("COMPARAÇÃO DOS MODELOS")
print("="*60)

# Tabela comparativa
comparison = pd.DataFrame({
    'Modelo': ['Nulo', 'Efeitos Fixos', 'Com Interação'],
    'AIC': [model_null.aic, model_fixed.aic, model_interaction.aic],
    'BIC': [model_null.bic, model_fixed.bic, model_interaction.bic],
    'LogLik': [model_null.llf, model_fixed.llf, model_interaction.llf]
})
print(comparison)

# Teste de razão de verossimilhança (Nulo vs Fixos)
lr_fixed = -2 * (model_null.llf - model_fixed.llf)
p_fixed = chi2.sf(lr_fixed, df=2)  # 2 parâmetros adicionados
print(f"\nTeste LRT (Nulo vs Fixos): LR = {lr_fixed:.3f}, p = {p_fixed:.6f}")

# Teste de razão de verossimilhança (Fixos vs Interação)
lr_interaction = -2 * (model_fixed.llf - model_interaction.llf)
p_interaction_lr = chi2.sf(lr_interaction, df=1)  # 1 parâmetro adicional
print(f"Teste LRT (Fixos vs Interação): LR = {lr_interaction:.3f}, p = {p_interaction_lr:.6f}")

# Melhor modelo
best_aic = comparison.loc[comparison['AIC'].idxmin(), 'Modelo']
print(f"\n✅ Melhor modelo segundo AIC: {best_aic}")

###### Grafico ##### Visualização da interação

import matplotlib.pyplot as plt
import numpy as np

# Gráfico de como a frequência afeta o faturamento
# para clientes baratos vs caros
fig, ax = plt.subplots(figsize=(10,6))

# Valores de preço (10º e 90º percentil)
p25 = df_invoice['logCustomerAvgPrice'].quantile(0.25)
p75 = df_invoice['logCustomerAvgPrice'].quantile(0.75)

freq_range = np.linspace(0, 5, 100)

# Previsões
pred_p25 = 4.542 + 0.001*freq_range + 0.120*p10 + 0.072*freq_range*p25
pred_p75 = 4.542 + 0.001*freq_range + 0.120*p90 + 0.072*freq_range*p75

ax.plot(freq_range, pred_p25, 'b-', label='Itens baratos (P10)')
ax.plot(freq_range, pred_p75, 'r-', label='Itens caros (P90)')
ax.set_xlabel('Frequência de compras (log)')
ax.set_ylabel('Faturamento esperado (log)')
ax.set_title('Interação: Efeito da frequência muda conforme o preço')
ax.legend()
plt.show()

## PERGUNTA 2 ###### 
###### o EFEITO DO NÚMERO DE ITENS NA FATURA SOBRE O FATURAMENTO VARIA ENTRE CLIENTES?? ######

model_slope = mixedlm(
    "logTotalAmount ~ NumItems + logFrequency * logCustomerAvgPrice",
    df_invoice,
    groups=df_invoice["CustomerID"],
    re_formula="~NumItems"  # Slope aleatório para NumItems
).fit()

print(model_slope.summary())

# Comparar com modelo sem slope aleatório
# Verificar se os modelos convergiram
print(f"\nModelo sem slope (Interação):")
print(f"  Log-Likelihood: {model_interaction.llf:.2f}")
print(f"  Convergiu: {model_interaction.converged}")

print(f"\nModelo COM slope aleatório:")
print(f"  Log-Likelihood: {model_slope.llf:.2f}")
print(f"  Convergiu: {model_slope.converged}")

# Calcular diferença
diff_llf = model_slope.llf - model_interaction.llf
print(f"\n📊 Diferença no Log-Likelihood: {diff_llf:.2f}")

if diff_llf > 0:
    print(f"✅ Modelo COM slope é MELHOR (LogLik maior em {diff_llf:.2f})")
else:
    print(f"❌ Modelo SEM slope é melhor")

if model_slope.aic < model_interaction.aic:
    print("✅ Modelo com slope aleatório é MELHOR")
else:
    print("❌ Modelo sem slope aleatório é SUFICIENTE")
    
### >> Sim, o efeito do numero de itens na fatura varia entre clientes ###





