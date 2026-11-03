# API de Previsão de Preços de Imóveis

API REST para estimar o preço de venda de imóveis em King County, Washington (EUA), usando características da propriedade e sua localização. O modelo de regressão apoia análises e decisões de compradores, vendedores, corretores e instituições financeiras; a estimativa não substitui uma avaliação profissional.

## Origem do modelo

O modelo foi desenvolvido no **Projeto Final do Módulo 1**. A API reutiliza a versão v1 do Gradient Boosting, treinada com o conjunto de imóveis de King County, e seus artefatos versionados:

- Modelo: [`models/v1/modelo_gradientBoosting_v1.pkl`](models/v1/modelo_gradientBoosting_v1.pkl)
- Scaler: [`models/v1/scaler_gradientBoosting_v1.pkl`](models/v1/scaler_gradientBoosting_v1.pkl)
- Metadados e métricas: [`models/v1/metricas_v1.json`](models/v1/metricas_v1.json)
- Documentação das métricas: [`models/v1/metricas_v1.md`](models/v1/metricas_v1.md)

O preço previsto é expresso em dólares americanos (USD). Na avaliação holdout 80/20 registrada para a v1, o modelo obteve MAE de aproximadamente **US$ 77.466,75**, RMSE de **US$ 142.419,42** e R² de **0,8658**. Essas métricas estimam o desempenho da avaliação anterior ao retreino final com toda a base; não são uma nova avaliação independente do artefato final.

## Tecnologias

Versões observadas no ambiente de desenvolvimento usado para validar a API:

| Tecnologia | Versão |
|---|---:|
| Python | 3.14.4 |
| FastAPI | 0.162.0 |
| Pydantic | 2.14.1 |
| Scikit-Learn | 1.9.0 |
| Uvicorn | 0.54.0 |
| NumPy | 2.5.4 |
| Pandas | 3.1.0 |
| Joblib | 1.6.0 |
| Matplotlib | 3.11.2 |
| Seaborn | 0.13.2 |
| Statsmodels | 0.15.0 |
| Jupyter | 1.1.1 |
| HTTPX (testes) | 0.28.1 |

As versões das bibliotecas listadas estão fixadas em [`requirements.txt`](requirements.txt) para tornar a instalação reproduzível. O Scikit-Learn foi alinhado à versão usada para serializar o modelo v1.

## Instalação e execução

Na raiz do repositório, crie e ative um ambiente virtual e instale as dependências:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Inicie o servidor local com Uvicorn:

```powershell
uvicorn main:app --reload
```

A API ficará disponível em `http://127.0.0.1:8000`. O ciclo de vida carrega e valida o modelo e o scaler na memória durante a inicialização; as requisições de inferência reutilizam esses artefatos.

> **Artefatos necessários:** confirme que `modelo_gradientBoosting_v1.pkl`, `scaler_gradientBoosting_v1.pkl` e `metricas_v1.json` estão em `models/v1/` antes de iniciar. Se precisar regenerá-los a partir do treinamento, execute a célula de exportação correspondente no notebook do Projeto Final do Módulo 1.

## Documentação da API

Com o servidor ativo, consulte o Swagger UI em [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs) ou o schema OpenAPI em [`http://127.0.0.1:8000/openapi.json`](http://127.0.0.1:8000/openapi.json).

### Monitoramento — `GET /health`

Retorna HTTP 200 e informa se a API está ativa e se modelo e scaler estão carregados em memória.

```powershell
curl.exe http://127.0.0.1:8000/health
```

### Previsão — `POST /predict`

O corpo deve conter as 23 features esperadas. Tipos numéricos são estritos, os campos extras não são aceitos e os valores devem respeitar as faixas documentadas no schema. Exemplo de imóvel:

```json
{
  "bedrooms": 3,
  "bathrooms": 2.5,
  "sqft_living": 1500,
  "sqft_lot": 5000,
  "floors": 1.5,
  "waterfront": 0,
  "view": 0,
  "condition": 3,
  "grade": 7,
  "sqft_basement": 0,
  "idade_imovel": 50,
  "foi_reformado": 0,
  "lat": 47.5,
  "long": -122.2,
  "zipcode_faixa_faixa_1": 1,
  "zipcode_faixa_faixa_2": 0,
  "zipcode_faixa_faixa_3": 0,
  "zipcode_faixa_faixa_4": 0,
  "zipcode_faixa_faixa_5": 0,
  "zipcode_faixa_faixa_6": 0,
  "zipcode_faixa_faixa_7": 0,
  "zipcode_faixa_faixa_8": 0,
  "zipcode_faixa_faixa_9": 0
}
```

Exemplo de chamada a partir do PowerShell:

```powershell
$body = Get-Content -Raw .\payload.json
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body $body
```

Em caso de sucesso, a resposta HTTP 200 segue este formato (o valor é apenas ilustrativo):

```json
{
  "preco_estimado": 475000.0,
  "versao_modelo": "v1"
}
```

#### Códigos HTTP

| Código | Significado |
|---|---|
| `200 OK` | Requisição processada com sucesso. |
| `400 Bad Request` | Corpo da requisição contém JSON malformado. |
| `422 Unprocessable Entity` | JSON válido, mas os campos não atendem ao schema ou às regras de validação. |
| `500 Internal Server Error` | Falha inesperada; os detalhes são registrados no log e uma mensagem genérica é devolvida. |

## Demonstração

- **Vídeo de demonstração:** (https://drive.google.com/drive/folders/1xRJjRlnKGtXn-6OWzVSWteN_Ivj6L4Fs?usp=sharing)
